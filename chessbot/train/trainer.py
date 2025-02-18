import argparse
import logging
import os

import sys
import random
import time
import yaml
import wandb
import importlib

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
from accelerate import Accelerator

from chessbot.data.dataset import ChessDataset
from chessbot.train.utils import WarmupLR, MetricsTracker
from chessbot.train.config import load_default_cfg
from chessbot.models.registry import ModelRegistry


class ChessTrainer:
    """Chess trainer for training a chess model with the ChessBot dataset

    Due to the size of the dataset, the training is split into rounds and epochs. Each round
    samples a subset of the training data, and then performs a specified number of epochs.
    """

    def __init__(self, config, model=None, load_model_from_config=False):
        config = OmegaConf.load(config) if isinstance(config, str) else config
        self.cfg = load_default_cfg()
        self.cfg = OmegaConf.merge(self.cfg, config)

        self.optimizer = None
        self.scheduler = None

        self.model = model
        load_model_from_config = True if model is None else load_model_from_config

        if load_model_from_config:
            self.load_model_from_config()
            self.initialize_model()

        if self.cfg.logging.wandb:
            wandb.init(project=self.cfg.logging.wandb_project)

        # Automatically use dated experiment directory
        self.cfg.train.output_dir = os.path.join(
            self.cfg.train.output_dir, f"{time.strftime('%Y-%m-%d_%H-%M')}-experiment"
        )
        self.checkpoint_dir = os.path.join(self.cfg.train.output_dir, "checkpoint")

        os.makedirs(self.cfg.train.output_dir, exist_ok=True)
        with open(os.path.join(self.cfg.train.output_dir, "config.yaml"), "w") as f:
            OmegaConf.save(self.cfg, f)

        self.latest_model_path = os.path.join(self.cfg.train.output_dir, "model_latest")
        self.best_model_path = os.path.join(self.cfg.train.output_dir, "model_best")

        self.stats = MetricsTracker()
        self.stats.add(
            [
                "train_loss",
                "train_ploss",
                "train_vloss",
                "val_loss",
                "val_ploss",
                "val_vloss",
            ]
        )

    def load_model_from_config(self):
        """
        Load the model dynamically. Prioritize 'model_file' if provided; otherwise, use 'model_hub'.
        """
        model_file = self.cfg['model'].get("model_file")
        model_hub_dir = self.cfg['model']['model_hub']

        if model_file:
            self._import_model_from_file(model_file)
        elif model_hub_dir:
            self._import_models_from_hub(model_hub_dir)
        else:
            raise ValueError(
                "No model_file or model_hub directory specified in the configurat.ion."
            )

    def _import_model_from_file(self, model_file):
        """
        Import a model from a user-specified file.
        """
        if not os.path.isfile(model_file):
            raise ValueError(f"Provided model file '{model_file}' does not exist.")

        module_name = os.path.splitext(os.path.basename(model_file))[0]

        self._import_module_from_path(module_name, model_file)

    def _import_models_from_hub(self, model_hub_dir):
        """
        Import all Python files from the model hub directory.
        """
        if not os.path.isdir(model_hub_dir):
            raise ValueError(f"Model hub directory '{model_hub_dir}' does not exist.")

        for filename in os.listdir(model_hub_dir):
            if filename.endswith(".py"):
                module_name = filename[:-3]
                module_path = os.path.join(model_hub_dir, filename)

                try:
                    self._import_module_from_path(module_name, module_path)
                except Exception as e:
                    print(f"Failed to load model file '{module_name}': {e}")

    def _import_module_from_path(self, module_name, module_path):
        """
        Helper function to import a module from a given path.
        """
        if module_name in sys.modules:
            print(f"Module '{module_name}' is already imported, skipping.")
            return  # Module is already imported; no need to re-import

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    def initialize_model(self):
        """
        Initialize the model using the registry.
        """
        model_name = self.cfg.model.name
        model_args = self.cfg.model.get('args', [])
        model_kwargs = self.cfg.model.get('kwargs', {})

        if not ModelRegistry.exists(model_name):
            raise ValueError(f"Model '{model_name}' is not registered.")

        ModelClass = ModelRegistry.get(model_name)

        self.model = ModelClass(*model_args, **model_kwargs)
        self.model.to(self.cfg.train.device)

    def build_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.train.lr
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

    @staticmethod
    def build_train_loader(cfg) -> DataLoader:
        train_path = os.path.join(cfg.dataset.data_path, "train")
        data = [pgn.path for pgn in os.scandir(train_path) if pgn.name.endswith(".pgn")]
        sampled_data = random.sample(data, cfg.dataset.size_train)
        dataset = ChessDataset(sampled_data, num_threads=cfg.dataset.num_threads)
        return DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)

    @staticmethod
    def build_val_loader(cfg) -> DataLoader:
        test_path = os.path.join(cfg.dataset.data_path, "test")
        data = [pgn.path for pgn in os.scandir(test_path) if pgn.name.endswith(".pgn")]
        sampled_data = random.sample(data, cfg.dataset.size_test)
        dataset = ChessDataset(sampled_data, num_threads=cfg.dataset.num_threads)
        return DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False)

    def run_validation(self):
        val_loader = self.build_val_loader(self.cfg)
        total_val_steps = len(val_loader)

        with tqdm(
            total=total_val_steps,
            desc="Validation",
            leave=False,
            dynamic_ncols=True,
            position=0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] {postfix}",
        ) as val_bar:
            self.model.eval()
            self.stats.reset(["val_loss", "val_ploss", "val_vloss"])

            with torch.no_grad():
                for state, action, result in val_loader:
                    state = state.float().to(self.cfg.train.device)
                    action = action.to(self.cfg.train.device)
                    result = result.float().to(self.cfg.train.device)

                    policy_output, value_output = self.model(state.unsqueeze(1))

                    policy_loss = self.model.policy_loss(policy_output.squeeze(), action)
                    value_loss = self.model.value_loss(value_output.squeeze(), result)

                    loss = policy_loss + value_loss

                    self.stats.update(
                        {
                            "val_loss": loss.item(),
                            "val_ploss": policy_loss.item(),
                            "val_vloss": value_loss.item(),
                        }
                    )

                    # Update the validation progress bar to display current loss.
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

    def training_round(self, train_loader, accelerator, progress_bar, round):
        """
        Perform a single training round.
        """
        self.model.train()
        best_val_loss = float('inf')
        val_loss = float('inf')

        for epoch in range(self.cfg.train.epochs):

            for iter, (state, action, result) in enumerate(train_loader):
                state = state.float()
                result = result.float()

                with accelerator.accumulate():
                    policy_output, value_output = self.model(state.unsqueeze(1))

                    policy_loss = self.model.policy_loss(policy_output.squeeze(), action)
                    value_loss = self.model.value_loss(value_output.squeeze(), result)
                    loss = policy_loss + value_loss

                    self.optimizer.zero_grad(set_to_none=True)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0
                        )

                    self.optimizer.step()

                    if self.scheduler:
                        self.scheduler.step()

                self.stats.update(
                    {
                        "train_loss": loss.item(),
                        "train_ploss": policy_loss.item(),
                        "train_vloss": value_loss.item(),
                    }
                )

                if self.cfg.logging.wandb:
                    wandb.log(
                        {
                            "train_loss": loss.item(),
                            "train_ploss": policy_loss.item(),
                            "train_vloss": value_loss.item(),
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
                        accelerator.save_model(self.model, self.best_model_path)
                        accelerator.save_state(output_dir=self.checkpoint_dir)

                # Update the single progress bar with detailed information
                progress_bar.set_postfix(
                    {
                        "Round": f"{round + 1}/{self.cfg.train.rounds}",
                        "Epoch": f"{epoch + 1}/{self.cfg.train.epochs}",
                        "Iter": f"{iter + 1}/{len(train_loader)}",
                        "Loss": f"{loss.item():.4f}",
                        "Val Loss": f"{val_loss:.4f}",
                    }
                )
                progress_bar.update(1)

            accelerator.save_model(self.model, self.latest_model_path)
            accelerator.save_state(output_dir=self.checkpoint_dir)

    def train(self):
        assert (
            self.model is not None
        ), "Model not initialized. Please set up your model before calling .train()"

        # Sets optimizer and scheduler if not already set
        if not self.optimizer:
            self.build_optimizer()

        accelerator = Accelerator(
            cpu=self.cfg.train.device == 'cpu',
            mixed_precision=self.cfg.train.amp,
            dynamo_backend='INDUCTOR' if self.cfg.train.compile else None,
        )

        # Each round samples new data from dataset and performs epochs
        for round_num in range(self.cfg.train.rounds):
            self.stats.reset(
                [
                    "train_loss",
                    "train_ploss",
                    "train_vloss",
                    "val_loss",
                    "val_ploss",
                    "val_vloss",
                ]
            )

            train_loader = self.build_train_loader(self.cfg)
            print(f"Loaded {len(train_loader.dataset.data)} positions")

            total_iters = len(train_loader) * self.cfg.train.epochs
            progress_bar = tqdm(
                desc="Epoch:",
                total=total_iters,
                leave=True,
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] {postfix}",
            )

            self.model, self.optimizer, train_loader, self.scheduler = (
                accelerator.prepare(
                    self.model, self.optimizer, train_loader, self.scheduler
                )
            )
            if self.cfg.train.checkpoint_dir and round_num == 0:
                accelerator.load_state(self.cfg.train.checkpoint_dir)

            self.training_round(train_loader, accelerator, progress_bar, round_num)

            # reduce memory spikes
            del train_loader

        progress_bar.close()


# if __name__ == "__main__":
#     import argparse

#     def parse_args():
#         parser = argparse.ArgumentParser(description="Chess Trainer")
#         parser.add_argument(
#             "--config",
#             type=str,
#             required=True,
#             help="Path to the YAML configuration file.",
#         )
#         parser.add_argument(
#             "--override",
#             nargs="*",
#             help="Override any config variable using dot notation, e.g., training.lr=0.001.",
#         )
#         return parser.parse_args()

#     def apply_overrides(config, overrides):
#         if overrides:
#             for override in overrides:
#                 keys, value = override.split("=", 1)
#                 keys = keys.split(".")
#                 sub_config = config
#                 for key in keys[:-1]:
#                     sub_config = sub_config.setdefault(key, {})
#                 sub_config[keys[-1]] = yaml.safe_load(
#                     value
#                 )  # Parse value as YAML for proper typing
#         return config

#     args = parse_args()

#     with open(args.config, 'r') as f:
#         config = yaml.safe_load(f)

#     config = apply_overrides(config, args.override)

#     trainer = ChessTrainer(config)

#     logging.info("Starting Chess Trainer...")
#     trainer.train()
#     logging.info("Training completed.")
