import logging
import os
import time
import random
from typing import List

from tqdm import tqdm
from omegaconf import OmegaConf
import wandb

import torch
from torch.utils.data import DataLoader

from timm.optim import create_optimizer_v2, list_optimizers

from accelerate import Accelerator

from chessbot.common import setup_logger, GREEN, RESET, DEFAULT_DATASET_DIR
from chessbot.data.dataset import HFChessDataset, ChessDataset

from chessbot.train.utils import WarmupLR, MetricsTracker
from chessbot.train.config import get_cfg
from chessbot.models.registry import ModelRegistry



class BaseChessTrainer:
    def __init__(self, config, model=None, load_model_from_config=False):
        # Logger and config
        self._logger = setup_logger("chessbot.trainer", level=logging.INFO)
        config = OmegaConf.load(config) if isinstance(config, str) else config
        self.cfg = OmegaConf.merge(get_cfg(), config)
        self._logger.info(f"Loaded configuration: \n {OmegaConf.to_yaml(self.cfg)}")

        # Setup output dirs
        if self.cfg.train.resume_from_checkpoint:
            if not self.cfg.train.output_dir:
                self.cfg.train.output_dir = os.path.dirname(self.cfg.train.checkpoint_dir)
        else:
            if not self.cfg.train.output_dir:
                self.cfg.train.output_dir = os.path.join(
                    './', f"{time.strftime('%Y-%m-%d_%H-%M')}-experiment"
                )

        self.model = model
        assert model is not None or self.cfg.model.name is not None, \
            "Model not provided in config or as an argument"

        # Paths for checkpoints
        self.latest_model_path = os.path.join(self.cfg.train.output_dir, "model_latest")
        self.best_model_path = os.path.join(self.cfg.train.output_dir, "model_best")
        self.checkpoint_dir = os.path.join(self.cfg.train.output_dir, "checkpoint")

        # Load model if requested
        if load_model_from_config:
            self.load_model_from_config()

        self._logger.info(f"Training with model: {self.model.__class__.__name__}")
        self._logger.info(self.model)

        # Accelerator
        self.accelerator = Accelerator(
            cpu=self.cfg.train.device == 'cpu',
            mixed_precision=self.cfg.train.amp,
            dynamo_backend='INDUCTOR' if self.cfg.train.compile else None,
        )

        # Optimizer & scheduler placeholders
        self.optimizer = None
        self.scheduler = None

        # Losses
        self.policy_loss = torch.nn.CrossEntropyLoss()
        self.value_loss = torch.nn.MSELoss()

        # Save config
        os.makedirs(self.cfg.train.output_dir, exist_ok=True)
        with open(os.path.join(self.cfg.train.output_dir, "config.yaml"), "w") as f:
            OmegaConf.save(self.cfg, f)

        self._logger.info(
            f"Saving training outputs to: {self.cfg.train.output_dir} - Resuming: {self.cfg.train.resume_from_checkpoint}"
        )

        # Weights & Biases
        if self.cfg.logging.wandb:
            resume = 'must' if self.cfg.logging.wandb_run_id else None
            wandb.init(
                project=self.cfg.logging.wandb_project,
                name=self.cfg.logging.wandb_run_name,
                id=self.cfg.logging.wandb_run_id,
                resume=resume,
            )

        # Metrics
        self.stats = MetricsTracker()
        self.stats.add(
            "train_loss", "train_ploss", "train_vloss",
            "val_loss", "val_ploss", "val_vloss",
        )

    def load_model_from_config(self):
        model_path = self.cfg.model.path
        model_name = self.cfg.model.name
        model_args = self.cfg.model.get('args', []) or []
        model_kwargs = self.cfg.model.get('kwargs', {}) or {}
        self.model = ModelRegistry.load_model(model_name, model_path, *model_args, **model_kwargs)
        self.model.to(self.cfg.train.device)
    
    def build_optimizer(self):
        optimizer_str = self.cfg.train.optimizer.lower()
        assert optimizer_str in list_optimizers(), f"Optimizer {optimizer_str} not valid - must be one of: {list_optimizers()}"
        
        if optimizer_str == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.cfg.train.lr
            )
        elif optimizer_str == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=self.cfg.train.lr, 
                momentum=0.9
            )
        else:
            self.optimizer = create_optimizer_v2(
                self.model.parameters(),
                opt=optimizer_str,
                lr=self.cfg.train.lr
            )

        self.scheduler = None

        if self.cfg.train.scheduler == "linear":
            min_scale = self.cfg.train.min_lr / self.cfg.train.lr
            scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=min_scale,
                total_iters=self.cfg.train.scheduler_iters,
            )
        elif self.cfg.train.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                self.cfg.train.scheduler_iters,
                self.cfg.train.min_lr,
            )

        if self.scheduler is not None and self.cfg.train.warmup_iters > 0:
            self.scheduler = WarmupLR(
                scheduler,
                init_lr=self.cfg.train.warmup_lr,
                num_warmup=self.cfg.train.warmup_iters,
                warmup_strategy=self.cfg.train.warmup_strategy,
            )

    def update_and_log(self, losses: dict, iter: int, train_or_val: str = "train"):
        self.stats.update(losses)
        cond = self.cfg.logging.wandb
        if train_or_val == "train":
            cond = cond and iter % self.cfg.logging.log_every == 0
        if cond:            
            avg_dict = {k: self.stats.get_average(k) for k in losses.keys()}
            wandb.log({**avg_dict, "iter": iter})
  
    def train_step(self, state, action, result):
        state = state.float()
        result = result.float()

        with self.accelerator.accumulate():
            policy_output, value_output = self.model(state.unsqueeze(1))

            policy_loss = self.policy_loss(policy_output.squeeze(), action)
            value_loss = self.value_loss(value_output.squeeze(), result)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

        return policy_loss, value_loss, loss

    def run_validation(self):
        val_loader = self.build_val_loader(self.cfg)

        with tqdm(
            total=1,
            desc="Validation",
            leave=False,
            dynamic_ncols=True,
            position=0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] {postfix}",
        ) as val_bar:
            self.model.eval()
            self.stats.reset("val_loss", "val_ploss", "val_vloss")

            with torch.no_grad():
                for state, action, result in val_loader:
                    state = state.float().to(self.cfg.train.device)
                    action = action.to(self.cfg.train.device)
                    result = result.float().to(self.cfg.train.device)

                    policy_output, value_output = self.model(state.unsqueeze(1))

                    policy_loss = self.policy_loss(policy_output.squeeze(), action)
                    value_loss = self.value_loss(value_output.squeeze(), result)

                    loss = policy_loss + value_loss

                    self.stats.update(
                        {
                            "val_loss": loss.item(),
                            "val_ploss": policy_loss.item(),
                            "val_vloss": value_loss.item(),
                        }
                    )
                    val_bar.set_postfix(
                        {
                            "Val Loss": f"{loss.item():.4f}",
                            "Val Ploss": f"{policy_loss.item():.4f}",
                            "Val Vloss": f"{value_loss.item():.4f}",
                        }
                    )
                    val_bar.update(1)

        return (
            self.stats.get_average('val_loss'),
            self.stats.get_average('val_ploss'),
            self.stats.get_average('val_vloss'),
        )
    

class HFChessTrainer(BaseChessTrainer):
    """Chess trainer for training a chess model using the ChessBot dataset supplied on HuggingFace.
    
    Due to the gigantic size the dataset uses streaming to avoid huge disk and RAM usage.
    """

    def __init__(self, config, model=None, load_model_from_config=False):
        super().__init__(config, model, load_model_from_config)

        self.train_loader = self.build_train_loader(self.cfg)
        self._logger.info(f"Loaded {self.train_loader.dataset} for training")
        
        self.progress_bar = tqdm(
            desc=f"{GREEN}Training{RESET}",
            leave=True,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt} [{elapsed}] {postfix}",
        )

    @staticmethod
    def build_train_loader(cfg) -> DataLoader:
        dataset = HFChessDataset('train', shuffle_buffer=cfg.dataset.shuffle_buffer)
        return DataLoader(dataset, batch_size=cfg.train.batch_size, num_workers=cfg.dataset.num_workers)

    @staticmethod
    def build_val_loader(cfg) -> DataLoader:
        dataset = HFChessDataset('test', shuffle_buffer=cfg.dataset.shuffle_buffer, num_test_samples=cfg.dataset.num_test_samples)
        return DataLoader(dataset, batch_size=cfg.train.batch_size, num_workers=cfg.dataset.num_workers)  
    
    def train(self):
        assert (
            self.model is not None
        ), "Model not initialized. Please set up your model before calling .train()"

        # Sets optimizer and scheduler from config if not already set. Allows users to set manually
        if not self.optimizer:
            self.build_optimizer()

        self.model, self.optimizer, self.train_loader, self.scheduler = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.scheduler
            )
        )
        if self.cfg.train.checkpoint_dir:
            self.accelerator.load_state(self.cfg.train.checkpoint_dir)

        # Run training
        self.model.train()
        best_val_loss = float('inf')
        val_loss = float('inf')
    
        for epoch in range(self.cfg.train.epochs):
    
            for iter, data in enumerate(self.train_loader):
                state, action, result = data[0], data[1], data[2]
                policy_loss, value_loss, loss = self.train_step(state, action, result)
                loss_dict = {
                    "train_loss": loss.item(),
                    "train_ploss": policy_loss.item(),
                    "train_vloss": value_loss.item(),
                }
                self.update_and_log(loss_dict, iter)

                # Validation
                if (
                    self.cfg.train.validation_every > 0
                    and iter % self.cfg.train.validation_every == 0
                    and iter > 0
                ):
                    val_loss, val_ploss, val_vloss = self.run_validation()
                    loss_dict = {
                        "val_loss": val_loss,
                        "val_ploss": val_ploss,
                        "val_vloss": val_vloss,
                    }
                    self.update_and_log(loss_dict, iter, train_or_val="val")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.accelerator.save_model(self.model, self.best_model_path, safe_serialization=False)   
                        self.accelerator.save_state(output_dir=self.checkpoint_dir)
                    
                    self.progress_bar.refresh()

                self.progress_bar.set_postfix(
                    {
                        "Epoch": f"{epoch + 1}/{self.cfg.train.epochs}",
                        "Iter": f"{iter + 1}",
                        "Loss": f"{loss.item():.4f}",
                        "Val Loss": f"{val_loss:.4f}",
                    }
                )
                self.progress_bar.update(1)

                # Reduce gpu power
                # time.sleep(0.075)

            self.accelerator.save_model(self.model, self.latest_model_path, safe_serialization=False)
            self.accelerator.save_state(output_dir=self.checkpoint_dir)

        self._logger.info(f"Training complete!")
        self.progress_bar.close()



class PGNChessTrainer(BaseChessTrainer):
    """Chess trainer for training a chess model with the ChessBot dataset

    Due to the size of the dataset, the training is split into rounds and epochs. Each round
    samples a subset of the training data, and then performs a specified number of epochs.
    """

    def __init__(self, config, model=None, load_model_from_config=False):
        super().__init__(config, model, load_model_from_config)

        # Optional data_path check (for filesystem dataset)
        if hasattr(self.cfg.dataset, 'data_path') and self.cfg.dataset.data_path is None:
            assert os.path.exists(DEFAULT_DATASET_DIR), \
                f"Dataset not found at {DEFAULT_DATASET_DIR} and no data path provided in config"
            self.cfg.dataset.data_path = DEFAULT_DATASET_DIR
        
        self.progress_bar = tqdm(
            desc=f"{GREEN}Training{RESET}",
            total=0,
            leave=True,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] {postfix}",
        )

    @staticmethod
    def build_train_loader(cfg) -> DataLoader:
        train_path = os.path.join(cfg.dataset.data_path, "train")
        data = [pgn.path for pgn in os.scandir(train_path) if pgn.name.endswith(".pgn")]
        sampled_data = random.sample(data, cfg.dataset.size_train)
        dataset = ChessDataset(sampled_data, num_processes=cfg.dataset.num_processes)
        return DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)

    @staticmethod
    def build_val_loader(cfg) -> DataLoader:
        test_path = os.path.join(cfg.dataset.data_path, "test")
        data = [pgn.path for pgn in os.scandir(test_path) if pgn.name.endswith(".pgn")]
        sampled_data = random.sample(data, cfg.dataset.size_test)
        dataset = ChessDataset(sampled_data, num_processes=cfg.dataset.num_processes)
        return DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False)

    def training_round(self, train_loader, round):
        """
        Perform a single training round.
        """
        self.model.train()
        best_val_loss = float('inf')
        val_loss = float('inf')

        for epoch in range(self.cfg.train.epochs):

            for iter, (state, action, result) in enumerate(train_loader):
                policy_loss, value_loss, loss = self.train_step(state, action, result)

                self.stats.update(
                    {
                        "train_loss": loss.item(),
                        "train_ploss": policy_loss.item(),
                        "train_vloss": value_loss.item(),
                    }
                )

                if self.cfg.logging.wandb and iter % self.cfg.logging.log_every == 0:
                    wandb.log(
                        {
                            "train_loss": self.stats.get_average('train_loss'), 
                            "train_ploss": self.stats.get_average('train_ploss'), 
                            "train_vloss": self.stats.get_average('train_vloss'), 
                            "iter": iter,
                        }   
                    )

                # Validation
                if (
                    self.cfg.train.validation_every > 0
                    and iter % self.cfg.train.validation_every == 0
                    and iter > 0
                ):
                    val_loss, val_ploss, val_vloss = self.run_validation()
                    self.stats.update(
                        {
                            "val_loss": val_loss,
                            "val_ploss": val_ploss,
                            "val_vloss": val_vloss,
                        }
                    )

                    if self.cfg.logging.wandb:
                        wandb.log(
                            {
                                "val_loss": val_loss,
                                "val_ploss": val_ploss,
                                "val_vloss": val_vloss,
                                "iter": iter,
                            }
                        )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.accelerator.save_model(self.model, self.best_model_path, safe_serialization=False)   
                        self.accelerator.save_state(output_dir=self.checkpoint_dir)
                    
                    self.progress_bar.refresh()

                self.progress_bar.set_postfix(
                    {
                        "Round": f"{round + 1}/{self.cfg.train.rounds}",
                        "Epoch": f"{epoch + 1}/{self.cfg.train.epochs}",
                        "Iter": f"{iter + 1}/{len(train_loader)}",
                        "Loss": f"{loss.item():.4f}",
                        "Val Loss": f"{val_loss:.4f}",
                    }
                )
                self.progress_bar.update(1)

            self.accelerator.save_model(self.model, self.latest_model_path, safe_serialization=False)
            self.accelerator.save_state(output_dir=self.checkpoint_dir)
    
    def train(self):
        assert (
            self.model is not None
        ), "Model not initialized. Please set up your model before calling .train()"

        # Sets optimizer and scheduler from config if not already set
        if not self.optimizer:
            self.build_optimizer()

        # Each round samples new data from dataset and performs epochs
        self._logger.info(f"Chess Trainer starting...")
        for round_num in range(self.cfg.train.rounds):
            self.stats.reset(
                "train_loss",
                "train_ploss",
                "train_vloss",
                "val_loss",
                "val_ploss",
                "val_vloss"
            )

            train_loader = self.build_train_loader(self.cfg)
            self._logger.info(f"Loaded {len(train_loader.dataset.data)} positions")


            total_iters = len(train_loader) * self.cfg.train.epochs
            self.progress_bar.total += total_iters
            self.progress_bar.refresh()

            self.model, self.optimizer, train_loader, self.scheduler = (
                self.accelerator.prepare(
                    self.model, self.optimizer, train_loader, self.scheduler
                )
            )
            if self.cfg.train.checkpoint_dir and round_num == 0:
                self.accelerator.load_state(self.cfg.train.checkpoint_dir)

            self.training_round(train_loader, round_num)

            # reduce memory spikes
            del train_loader

        self._logger.info(f"Training complete!")
        self.progress_bar.close()


def train_fn_local(config_path: str, override: List[str] = None):
    """ Used in `chessbot train` cli
    Load a YAML configuration file, apply any quick overrides, train the model.

    Args:
        config_path (str): Path to the YAML configuration file.
        override (List[str], optional): List of overrides in dot notation,
                                        e.g., ["training.lr=0.001", "model.hidden_size=256"].
    """
    config = OmegaConf.load(config_path)

    # Create a config from the dotlist and merge it if present
    if override:
        override_conf = OmegaConf.from_dotlist(override)
        config = OmegaConf.merge(config, override_conf)

    trainer = PGNChessTrainer(config, load_model_from_config=True)
    trainer.train()


def train_fn_hf(config_path: str, override: List[str] = None):
    """ Used in `chessbot train` cli
    Load a YAML configuration file, apply any quick overrides, train the model.

    Args:
        config_path (str): Path to the YAML configuration file.
        override (List[str], optional): List of overrides in dot notation,
                                        e.g., ["training.lr=0.001", "model.hidden_size=256"].
    """
    config = OmegaConf.load(config_path)

    # Create a config from the dotlist and merge it if present
    if override:
        override_conf = OmegaConf.from_dotlist(override)
        config = OmegaConf.merge(config, override_conf)

    trainer = HFChessTrainer(config, load_model_from_config=True)
    trainer.train()