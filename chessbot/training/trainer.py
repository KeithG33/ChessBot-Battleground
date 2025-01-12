import argparse
import logging
import os
import time
import random
import yaml
import wandb
import importlib
from collections import OrderedDict
import attridict
import torch
from torch.utils.data import DataLoader

from chessbot.utils.utils import RunningAverage
from chessbot.data.dataset import ChessDataset
from chessbot.training.warmup import WarmupLR
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

    def __init__(self, config):
        self.load_config(config)
        self.import_model_from_config()
        self.initialize_model()

    def load_config(self, config):
        self.config = attridict(config)

        # Main stuffs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_save_path = self.config.model.save_path
        self.pgn_dir_train = self.config.dataset.train_path
        self.pgn_dir_test = self.config.dataset.test_path
        self.dataset_size_train = self.config.dataset.batchsize_train
        self.dataset_size_test = self.config.dataset.batchsize_test
        self.batch_size = self.config.training.batchsize
        self.num_rounds = self.config.training.num_rounds
        self.log_every = self.config.logging.log_every
        self.validation_every = self.config.logging.validation_every
        self.num_threads = self.config.dataset.num_threads
        self.wandb_enabled = self.config.logging.wandb

        if self.wandb_enabled:
            wandb.init(project=self.config.logging.wandb_project)

    def import_model_from_config(self):
        """
        Dynamically import the model file specified in the configuration.
        The model_hub directory contains model files, and the model_class can refer to
        either the file name (without .py) or the class name within the file.
        """
        model_hub_dir = self.config['model']['model_hub']
        model_class_name = self.config['model']['model_class']

        # Ensure the model hub directory exists
        if not os.path.isdir(model_hub_dir):
            raise ValueError(f"Model hub directory '{model_hub_dir}' does not exist.")

        # Search for a matching file or module in the model hub
        for filename in os.listdir(model_hub_dir):
            if filename.endswith(".py"):
                module_name = filename[:-3]  # Strip .py
                module_path = os.path.join(model_hub_dir, filename)

                # Import the module dynamically
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Check if the model class exists in the module
                if hasattr(module, model_class_name):
                    return  # Model class successfully imported

        raise ImportError(
            f"Model class '{model_class_name}' not found in any file within '{model_hub_dir}'."
        )

    def initialize_model(self):
        model_name = self.config['model']['name']
        model_args = self.config['model'].get('args', [])
        model_kwargs = self.config['model'].get('kwargs', {})

        # Retrieve the model class from the registry
        ModelClass = ModelRegistry.get(model_name)

        # Initialize the model
        self.model = ModelClass(*model_args, **model_kwargs)
        self.model.to(self.device)

    def build_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.training.lr
        )
        self.scheduler = None

        if self.config.training.scheduler == "linear":
            min_scale = self.config.training.min_lr / self.config.training.lr
            scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=min_scale,
                total_iters=self.config.training.scheduler_iters,
            )
        elif self.config.training.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                self.config.training.scheduler_iters,
                self.config.training.min_lr,
            )

        if scheduler is not None:
            self.scheduler = WarmupLR(
                scheduler,
                init_lr=self.config.training.warmup_lr,
                num_warmup=self.config.training.warmup_iters,
                warmup_strategy=self.config.training.warmup_strategy,
            )

        if os.path.exists(self.model_save_path):
            checkpoint = torch.load(self.model_save_path)
            self.model.load_state_dict(checkpoint)
            print("Loaded model from checkpoint")

    def run_validation(self, stats: RunningAverage):
        test_data = [
            pgn.path
            for pgn in os.scandir(self.pgn_dir_test)
            if pgn.name.endswith(".pgn")
        ]
        sampled_test_data = random.sample(test_data, self.dataset_size_test)
        test_dataset = ChessDataset(sampled_test_data, num_threads=self.num_threads)
        val_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        self.model.eval()
        stats.reset(["val_loss", "val_ploss", "val_vloss"])

        with torch.no_grad():
            for state, action, result in val_loader:
                state = state.float().to(self.device)
                action = action.to(self.device)
                result = result.float().to(self.device)

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
    
    def training_round(self, train_loader, accelerator):
        stats = RunningAverage()
        stats.add(
            [
                "train_loss",
                "val_loss",
                "train_ploss",
                "train_vloss",
                "val_ploss",
                "val_vloss",
            ]
        )
        best_val_loss = float('inf')

        self.model.train()

        for i, (state, action, result) in enumerate(train_loader):
            state = state.float()
            # action = action
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

            if self.wandb_enabled and i % self.log_every == 0 and i > 0:
                wandb.log(
                    {
                        "train_loss": stats.get_average('train_loss'),
                        "train_ploss": stats.get_average('train_ploss'),
                        "train_vloss": stats.get_average('train_vloss'),
                        "lr": self.scheduler.get_last_lr()[0] if self.scheduler else self.config.training.lr,
                        "iter": i,
                    }
                )

            if i % self.validation_every == 0 and i > 0:
                val_loss, val_ploss, val_vloss = self.run_validation(stats)

                if self.wandb_enabled:
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
                    torch.save(self.model.state_dict(), self.model_save_path)
                    accelerator.save_model(self.model, self.model_save_path)

    def train(self):
        train_data = [
            pgn.path
            for pgn in os.scandir(self.pgn_dir_train)
            if pgn.name.endswith(".pgn")
        ]

        accelerator = Accelerator(
            cpu = self.device == 'cpu',
            mixed_precision = self.config.training.amp,
            dynamo_backend = 'INDUCTOR' if self.config.training.compile else None,
        )
        accelerator.device = self.device

        for round_num in range(self.num_rounds):
            print(f"Starting round {round_num}")
            sampled_train_data = random.sample(train_data, self.dataset_size_train)
            train_dataset = ChessDataset(sampled_train_data, self.num_threads)
            train_loader = DataLoader(
                train_dataset, self.batch_size, shuffle=True
            )

            self.model, self.optimizer, train_loader, self.scheduler = accelerator.prepare(
                self.model, self.optimizer, train_loader, self.scheduler
            )

            self.training_round(train_loader, accelerator)

            del train_dataset, train_loader



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
