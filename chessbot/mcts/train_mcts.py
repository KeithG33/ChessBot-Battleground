from pathlib import Path
import os
import shutil
import time
from tqdm import tqdm
import wandb
import yaml

import torch
from torch.multiprocessing import set_start_method, Pool

from timm.optim import create_optimizer_v2, list_optimizers

from chessbot.mcts.train_utils import (
    SelfPlayMultiProcManager,
    run_training_epoch,
    run_play_match,
)
from chessbot.common import setup_logger


class Config:
    MODEL_CLASS = 'sgu_chessbot'  # Must be registered in the model registry.
    MODEL_ARGS = []  # Args to pass to the model constructor
    MODEL_KWARGS = {}  # Kwargs to pass to the model constructor

    # Optional pretrained weights
    MODEL_WEIGHTS = (
        '/home/kage/chess_workspace/ChessBot-Battleground/models/sgu_chessbot/2025-02-19_18-51-experiment/model_latest/model.pt'
    )

    # Training
    TRAIN_PGN_DIR: str = "/home/kage/chess_workspace/PGN_dataset/Dataset/train"
    TRAIN_TOTAL_GAMES = 1000  # Num of selfplay games before finishing training
    TRAIN_FREQ: int = 1  # Train ever TRAIN_FREQ games
    TRAIN_OUTPUT_DIR: str = '/home/kage/chess_workspace/ChessBot-Battleground/models/sgu_chessbot/mcts_training/'

    # Supervised training
    TRAIN_WITH_EXPERT = False  # Whether to include expert data in training
    TRAIN_EPOCHS: int = 1  # Num supervised train epochs
    TRAIN_DATASET_SIZE: int = 10_000_000  # Num positions in dataset
    TRAIN_EXPERT_SIZE = 1  # Num files of expert data to sample
    TRAIN_EXPERT_RATIO = 0.1  # Min percent of expert data 
    TRAIN_BATCH_SIZE: int = 2048
    TRAIN_OPTIMIZER = 'adamw'
    TRAIN_LR = 0.0001
    TRAIN_MIN_LR = 0.00001
    TRAIN_SCHEDULER = 'cosine'
    TRAIN_SCHEDULER_ITERS = 1_000_000
    TRAIN_WARMUP_LR = 0.00001
    TRAIN_WARMUP_ITERS = 1000

    # Selfplay
    SELFPLAY_PARALLEL: int = 1  # Num parallel selfplay processes (each running one game)
    SELFPLAY_SIMS: int = 100  # Num sims per move in mcts selfplay
    SELFPLAY_BUFFER_SIZE: int = 1_000_000  # Size of replay buffer queue

    # Match evaluation
    MATCH_ROUNDS: int = 3  # Best-of-MATCH_ROUNDS to decide best model
    MATCH_WINRATE: float = 0.55  # Required winrate to decide best model
    MATCH_SIMS: int = 25  # Num sims per move in MCTS match
    MATCH_PROCESSES: int = 2  # Num parallel match processes (each running one game)

    LOG_WANDB = False  # Whether to log to wandb


class MCTSTrainer:
    """
    Trainer class encapsulating the self-play, training, and match routines for A0-style training.

    In the main loop, the trainer starts a self-play process to continuously generate data. Once
    the cfg.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = setup_logger('MCTS-Trainer')

        self.mp_manager = SelfPlayMultiProcManager(cfg.SELFPLAY_BUFFER_SIZE)

        self.next_train = cfg.TRAIN_FREQ
        self.last_game_count = 0
        self.training = True

        # Setup output directory. Dump config to yaml file
        os.makedirs(cfg.TRAIN_OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(cfg.TRAIN_OUTPUT_DIR, 'config.yaml'), 'w') as f:
            yaml.dump(cfg.__dict__, f)

        # Save the initial model to current and best model paths
        self.setup_train_paths()


    def setup_train_paths(self):        
        self.cfg.MODEL_CURR_PATH = os.path.join(self.cfg.TRAIN_OUTPUT_DIR, 'model_latest.pt')
        self.cfg.MODEL_BEST_PATH = os.path.join(self.cfg.TRAIN_OUTPUT_DIR, 'model_best.pt')
        self.cfg.TRAIN_STATE_PATH = os.path.join(self.cfg.TRAIN_OUTPUT_DIR, 'train_state.pt')
        
        if not self.cfg.MODEL_WEIGHTS or not os.path.exists(self.cfg.MODEL_WEIGHTS):
            from chessbot.models import MODEL_REGISTRY
            model = MODEL_REGISTRY.load_model(
                self.cfg.MODEL_CLASS,
                init_args=self.cfg.MODEL_ARGS,
                init_kwargs=self.cfg.MODEL_KWARGS,
            )
            torch.save(model.state_dict(), self.cfg.MODEL_CURR_PATH)
            torch.save(model.state_dict(), self.cfg.MODEL_BEST_PATH)
        else:
            shutil.copy(self.cfg.MODEL_WEIGHTS, self.cfg.MODEL_CURR_PATH)
            shutil.copy(self.cfg.MODEL_WEIGHTS, self.cfg.MODEL_BEST_PATH)

    def start_selfplay(self):
        """Starts the self-play processes using multiprocessing."""
        self.mp_manager.start_self_play_process(
            self.cfg,
            self.mp_manager.shutdown_event,
            self.mp_manager.global_game_counter,
            self.mp_manager.file_lock,
            self.mp_manager.shared_replay_buffer
        )

    def save_model_if_better(self, match_score_dict, curr_best_score, curr_best_wins):
        tmp_best_model_state = None
        if match_score_dict['score'] > (
            self.cfg.MATCH_WINRATE * 2 * self.cfg.MATCH_ROUNDS
        ):
            self.logger.info("New model wins the match!")
            if match_score_dict['score'] > curr_best_score:
                curr_best_score = match_score_dict['score']
                curr_best_wins = match_score_dict['wins']
                tmp_best_model_state = torch.load(self.cfg.MODEL_CURR_PATH)
            elif (
                match_score_dict['score'] == curr_best_score
                and match_score_dict['wins'] > curr_best_wins
            ):
                curr_best_wins = match_score_dict['wins']
                tmp_best_model_state = torch.load(self.cfg.MODEL_CURR_PATH)
        return tmp_best_model_state

    def train_loop(self):
        """
        Main loop that manages the background self-play process and when to trigger the
        train_and_match process.
        """
        # pbar = tqdm(total=self.cfg.TRAIN_TOTAL_GAMES, desc="Self-Play Games:")

        # Make it pretty with green text
        desc = "\033[1;32mSelf-Play Progress\033[0m"
        pbar = tqdm(total=self.cfg.TRAIN_TOTAL_GAMES, desc=desc, initial=0)
        pbar.set_postfix({'Buffer': 0})
        
        while self.training:
            # Update progress bar based on games played
            game_count = self.mp_manager.global_game_counter.count
            games_completed = game_count - self.last_game_count
            self.last_game_count = game_count
            pbar.update(games_completed)

            buffer_len = self.mp_manager.shared_replay_buffer.get_state()['curr_length']
            pbar.set_postfix({'Data': buffer_len})
        
            # Check for completion
            if game_count >= self.cfg.TRAIN_TOTAL_GAMES:
                self.mp_manager.shutdown_event.set()
                self.training = False

            # Trigger a training round
            if game_count >= self.next_train:
                self.logger.info("Waiting for self-play games to finish...")
                self.mp_manager.shutdown_event.set()
                self.mp_manager.join_process()
                self.mp_manager.shutdown_event.clear()

                self.logger.info(
                    "Self-play paused. Starting training..."
                )
           
                self.train_and_match()
                self.next_train += self.cfg.TRAIN_FREQ

                self.logger.info("Training Complete. Restarting self-play process...")
                self.start_selfplay()

            # No need to check at hyper speed
            time.sleep(0.5)

        self.mp_manager.join_process()

    def train_and_match(self):
        """
        Supervised training and match evaluation to determine if the model is better. Each training epoch is
        followed by a battle against the previous best model.
        """
        curr_best_score = 0
        curr_best_wins = 0
        best_model_state = None

        for epoch in range(self.cfg.TRAIN_EPOCHS):
            # Train one epoch. Use separate process to ensure memory cleanup afterwards
            with Pool(1) as pool:
                stats = pool.apply(run_training_epoch, (self.mp_manager.shared_replay_buffer, self.cfg))

            self.logger.info(f"T")

            # Compare current model against best model with matches
            match_score_dict = run_play_match(
                self.cfg,
                self.cfg.MODEL_CURR_PATH,
                self.cfg.MODEL_BEST_PATH,
                self.cfg.MATCH_ROUNDS,
                self.mp_manager.file_lock,
                num_sims=self.cfg.MATCH_SIMS,
                num_processes=self.cfg.MATCH_PROCESSES,
            )
            self.logger.info(f"Match scoring: {match_score_dict}")
            if self.cfg.LOG_WANDB:
                wandb.log(match_score_dict)

            # Save model as current model if match score meets the winrate criteria (best model saved at end)
            tmp_best_model_state = self.save_model_if_better(
                match_score_dict, curr_best_score, curr_best_wins
            )

            if tmp_best_model_state is not None:
                curr_best_score = match_score_dict['score']
                curr_best_wins = match_score_dict['wins']
                best_model_state = tmp_best_model_state

            self.logger.info(
                f"Epoch {epoch} - Loss: {stats.get_average('loss')} | "
                f"Policy Loss: {stats.get_average('policy_loss')} | "
                f"Value Loss: {stats.get_average('value_loss')}"
            )
            if self.cfg.LOG_WANDB:
                wandb.log(
                {
                    "epoch_loss": stats.get_average('loss'),
                    "epoch_ploss": stats.get_average('policy_loss'),
                    "epoch_vloss": stats.get_average('value_loss'),
                }
            )

        # Save the best model from the matches, if improved
        if best_model_state is not None:
            torch.save(best_model_state, self.cfg.MODEL_BEST_PATH)
            self.mp_manager.selfplay_buffer_proxy.clear()

    def run(self):
        """
        Starts the self-play and training loop.
        """
        self.start_selfplay()
        self.train_loop()


if __name__ == "__main__":
    set_start_method('spawn', force=True)
    cfg = Config()
    trainer = MCTSTrainer(cfg)
    trainer.run()
