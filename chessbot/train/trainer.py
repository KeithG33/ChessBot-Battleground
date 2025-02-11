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

from chessbot.data.dataset import ChessDataset
from chessbot.train.utils import WarmupLR, RunningAverage
from chessbot.train.config import get_config, load_config
from chessbot.models.registry import ModelRegistry

from accelerate import Accelerator


class ChessTrainer:
    """Chess trainer for training a chess model using the chess dataset
    
    NOTE: The dataloading here is a little weird, and specific to my levels of home compute. 
          Users will likely want to write their own training code for their own setup.

          The training setup here uses 'rounds' and 'epochs'. Each 'round' samples a subset of the
          training data, and will perform a specified number of 'epochs' before sampling again to
          start a new round

          This is reasonable because ~5 minutes of dataloading can load >100 million
          positions on my AMD Epyc 7402 (using 20 processes), around 70Gb of RAM
    """

    def __init__(self, config, model=None, load_model_from_config=False):
        self.cfg = get_config()
        config = load_config(config) if isinstance(config, str) else config
        self.cfg |= config

        self.model = model
        if load_model_from_config:
            self.load_model_from_config()
            self.initialize_model()

        if self.cfg.logging.wandb:
            wandb.init(project=self.cfg.logging.wandb_project)

        # Automatically use dated experiment directory
        self.cfg.train.output_dir = os.path.join(
            self.cfg.train.output_dir, f"{time.strftime('%Y-%m-%d_%H-%M')}-experiment"
        )

        os.makedirs(self.cfg.train.output_dir, exist_ok=True)
        with open(os.path.join(self.cfg.train.output_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(self.cfg, f)

        self.latest_model_path = os.path.join(self.cfg.train.output_dir, "model_latest.pth")
        self.best_model_path = os.path.join(self.cfg.train.output_dir, "model_best.pth")
            
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
            raise ValueError("No model_file or model_hub directory specified in the configurat.ion.")

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

    def build_train_loader(self) -> DataLoader:
        train_path = os.path.join(self.cfg.dataset.data_path, "train")
        data = [
            pgn.path
            for pgn in os.scandir(train_path)
            if pgn.name.endswith(".pgn")
        ]
        sampled_data = random.sample(data, self.cfg.dataset.size_train)
        dataset = ChessDataset(sampled_data, num_threads=self.cfg.dataset.num_threads)
        return DataLoader(
            dataset, batch_size=self.cfg.train.batch_size, shuffle=True
        )

    def build_val_loader(self) -> DataLoader:
        test_path = os.path.join(self.cfg.dataset.data_path, "test")
        data = [
            pgn.path
            for pgn in os.scandir(test_path)
            if pgn.name.endswith(".pgn")
        ]
        sampled_data = random.sample(data, self.cfg.dataset.size_test)
        dataset = ChessDataset(sampled_data, num_threads=self.cfg.dataset.num_threads)
        return DataLoader(
            dataset, batch_size=self.cfg.train.batch_size, shuffle=False
        )

    def run_validation(self, stats: RunningAverage):
        val_loader = self.build_val_loader()

        self.model.eval()
        stats.reset(["val_loss", "val_ploss", "val_vloss"])

        with torch.no_grad():
            for state, action, result in val_loader:
                state = state.float().to(self.cfg.train.device)
                action = action.to(self.cfg.train.device)
                result = result.float().to(self.cfg.train.device)

                policy_output, value_output = self.model(state.unsqueeze(1))

                policy_loss = self.model.policy_loss(policy_output.squeeze(), action)
                value_loss = self.model.value_loss(value_output.squeeze(), result)

                loss = policy_loss + value_loss

                stats.update(
                    {
                        "val_loss": loss.item(),
                        "val_ploss": policy_loss.item(),
                        "val_vloss": value_loss.item(),
                    }
                )

        return (
            stats.get_average('val_loss'),
            stats.get_average('val_ploss'),
            stats.get_average('val_vloss'),
        )
    
    def training_round(self, train_loader, accelerator, progress_bar, stats):
        """
        Perform a single training round.
        """
        self.model.train()
        best_val_loss = float('inf')

        for epoch in range(self.cfg.train.epochs):
    
            for i, (state, action, result) in enumerate(train_loader):
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
                        accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.optimizer.step()

                    if self.scheduler:
                        self.scheduler.step()

                stats.update(
                    {
                        "train_loss": loss.item(),
                        "train_ploss": policy_loss.item(),
                        "train_vloss": value_loss.item(),
                    }
                )

                progress_bar.set_postfix(
                    {
                        "train_loss": stats.get_average("train_loss"),
                        "val_loss": stats.get_average("val_loss"),
                        "epoch": f"{epoch + 1}/{self.cfg.train.epochs}",
                        "iter": f"{i + 1}/{len(train_loader)}",
                    }
                )


                if i % self.cfg.train.validation_every == 0 and i > 0:
                    val_loss, val_ploss, val_vloss = self.run_validation(stats)
                    stats.update(
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
                                "iter": i,
                            }
                        )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        accelerator.save_model(self.model, self.best_model_path)
            
            accelerator.save_model(self.model, self.latest_model_path)

    def train(self):
        assert self.model is not None, "Model not initialized"

        # Sets optimizer and scheduler
        self.build_optimizer()

        accelerator = Accelerator(
            cpu=self.cfg.train.device == 'cpu',
            mixed_precision=self.cfg.train.amp,
            dynamo_backend='INDUCTOR' if self.cfg.train.compile else None,
        )

        progress_bar = tqdm(
            desc="Training Progress",
            leave=True,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] {postfix}",
        )

        # Each round samples new data from dataset and performs epochs
        for round_num in range(self.cfg.train.rounds):
            progress_bar.set_description(f"Round {round_num + 1}/{self.cfg.train.rounds}")

            stats = RunningAverage()
            stats.add(["train_loss", "val_loss", "train_ploss", "train_vloss", "val_ploss", "val_vloss"])

            train_loader = self.build_train_loader()

            print(f"Loaded {len(train_loader.dataset.data)} positions")

            self.model, self.optimizer, train_loader, self.scheduler = accelerator.prepare(
                self.model, self.optimizer, train_loader, self.scheduler
            )
            
            self.training_round(train_loader, accelerator, progress_bar, stats)

            # reduce memory spikes
            del train_loader

        progress_bar.close()


if __name__ == "__main__":
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(description="Chess Trainer")
        parser.add_argument(
            "--config",
            type=str,
            required=True,
            help="Path to the YAML configuration file.",
        )
        parser.add_argument(
            "--override",
            nargs="*",
            help="Override any config variable using dot notation, e.g., training.lr=0.001.",
        )
        return parser.parse_args()

    def apply_overrides(config, overrides):
        if overrides:
            for override in overrides:
                keys, value = override.split("=", 1)
                keys = keys.split(".")
                sub_config = config
                for key in keys[:-1]:
                    sub_config = sub_config.setdefault(key, {})
                sub_config[keys[-1]] = yaml.safe_load(
                    value
                )  # Parse value as YAML for proper typing
        return config

    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config = apply_overrides(config, args.override)

    trainer = ChessTrainer(config)

    logging.info("Starting Chess Trainer...")
    trainer.train()
    logging.info("Training completed.")
