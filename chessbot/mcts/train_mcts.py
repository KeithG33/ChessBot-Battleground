from pathlib import Path
import os
import shutil
import time
import torch
from tqdm import tqdm
from torch.multiprocessing import set_start_method, Pool
import wandb

from chessbot.mcts.train_utils import SelfPlayMultiProcManager, run_training_epoch, run_duel
from chessbot.common import setup_logger


class Config:
    MODEL_PATH = Path('/home/kage/chess_workspace/chessAttnMixer20.pt')
    CURR_MODEL_PATH = MODEL_PATH.parent / ('MCTS_' + MODEL_PATH.name)
    BEST_MODEL_PATH = MODEL_PATH.parent / ('bestMCTS_' + MODEL_PATH.name)  
    PGN_DIR: str = "/home/kage/chess_workspace/PGN_dataset/Dataset/train"
    
    # Training
    TRAIN_TOTAL_GAMES = 1000 # Num of selfplay games before finishing training
    TRAIN_FREQ: int = 100 # Num selfplay games between trains
    
    # Supervised training
    TRAIN_WITH_EXPERT = False # Whether to include expert data in training
    TRAIN_EPOCHS: int = 10 # Num supervised train epochs
    TRAIN_DATASET_SIZE: int = 2 # Num files of expert data 
    TRAIN_EXPERT_SIZE = 1 # Num files of expert data to sample
    TRAIN_EXPERT_RATIO = 0.1 # Min percent of expert data
    TRAIN_BATCH_SIZE: int = 1024 

    # Selfplay
    SELFPLAY_PARALLEL: int = 19 # Num parallel selfplay processes (each running one game)
    SELFPLAY_SIMS: int = 500 # Num sims per move in mcts selfplay
    SELFPLAY_BUFFER_SIZE: int = 100_000 # Size of replay buffer queue
   
    # Dueling
    DUEL_ROUNDS: int = 11 # Best-of-DUEL_ROUNDS to decide best model
    DUEL_WINRATE: float = 0.55 # Required winrate to decide best model
    DUEL_SIMS: int = 250 # Num sims per move in mcts dueling
    DUEL_PROCESSES: int = 11 # Num parallel duel processes (each running one game) 
       

class MCTSTrainer:
    """ 
    Trainer class encapsulating the self-play, training, and dueling routines for A0-style training.

    In the main loop, the trainer starts a self-play process to continuously generate data. Once
    the cfg.


    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = setup_logger('MCTS-Trainer')
        
        # Save the initial model to both current and best model paths
        self.save_initial_model()

        self.mp_manager = SelfPlayMultiProcManager(cfg.SELFPLAY_BUFFER_SIZE)

        self.next_train = cfg.TRAIN_PERIOD
        self.last_game_count = 0
        self.training = True
    
    def save_initial_model(self):
        shutil.copy(self.cfg.MODEL_PATH, self.cfg.CURR_MODEL_PATH)
        shutil.copy(self.cfg.MODEL_PATH, self.cfg.BEST_MODEL_PATH)

    def start_selfplay(self):
        """Starts the self-play processes using multiprocessing."""
        self.mp_manager.start_self_play_process(
            self.cfg,
            self.mp_manager.shutdown_event,
            self.mp_manager.global_game_counter,
            self.mp_manager.file_lock,
            self.mp_manager.shared_replay_buffer
        )

    def save_model_if_better(self, duel_score_dict, curr_best_score, curr_best_wins):
        tmp_best_model_state = None
        if duel_score_dict['score'] > (self.cfg.DUEL_WINRATE * 2 * self.cfg.DUEL_ROUNDS):
            self.logger.info("New model wins the duel!")
            if duel_score_dict['score'] > curr_best_score:
                curr_best_score = duel_score_dict['score']
                curr_best_wins = duel_score_dict['wins']
                tmp_best_model_state = torch.load(self.cfg.CURR_MODEL_PATH)
            elif (
                    duel_score_dict['score'] == curr_best_score and
                    duel_score_dict['wins'] > curr_best_wins
                ):
                curr_best_wins = duel_score_dict['wins']
                tmp_best_model_state = torch.load(self.cfg.CURR_MODEL_PATH)
        return tmp_best_model_state
    
    def train_loop(self):
        """
        Main loop that manages the background self-play process and when to trigger the
        train_and_duel process.
        """
        pbar = tqdm(total=self.cfg.TRAIN_TOTAL_GAMES, desc="Self-play Game Count")
        while self.training:
            # Update progress bar based on games played
            game_count = self.mp_manager.global_game_counter.count
            games_completed = game_count - self.last_game_count
            self.last_game_count = game_count
            pbar.update(games_completed)

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

                self.logger.info("Self-play complete. Saving replay buffer and starting training...")
                self.train_and_duel()
                self.next_train += self.cfg.TRAIN_FREQ

                self.logger.info("Training Complete. Restarting self-play process...")
                self.start_selfplay()

            # No need to check at hyper speed
            time.sleep(0.5)

        self.mp_manager.join_process()

    def train_and_duel(self):
        """
        Supervised training and dueling to determine if the model is better. Each training epoch is
        is followed by a battle against the previous best model.
        """
        curr_best_score = 0
        curr_best_wins = 0
        best_model_state = None

        for epoch in range(self.cfg.TRAIN_EPOCHS):
            # Train one epoch. Use separate process to ensure memory cleanup afterwards
            with Pool(1) as pool:
                stats = pool.apply(run_training_epoch, (self.mp_manager, self.cfg))

            # Run dueling process to compare current model against best model
            duel_score_dict = run_duel(
                self.cfg.CURR_MODEL_PATH,
                self.cfg.BEST_MODEL_PATH,
                self.cfg.DUEL_ROUNDS,
                self.mp_manager.file_lock,
                num_sims=self.cfg.DUEL_SIMS,
                num_processes=self.cfg.DUEL_PROCESSES,
            )
            self.logger.info(f"Duel scoring: {duel_score_dict}")
            wandb.log(duel_score_dict)

            # Save model as current model if duel score meets the winrate criteria (best model saved at end)
            tmp_best_model_state = self.save_model_if_better(duel_score_dict, curr_best_score, curr_best_wins)
            
            if tmp_best_model_state is not None:
                curr_best_score = duel_score_dict['score']
                curr_best_wins = duel_score_dict['wins']
                best_model_state = tmp_best_model_state

            self.logger.info(
                f"Epoch {epoch} - Loss: {stats.get_average('loss')} | "
                f"Policy Loss: {stats.get_average('policy_loss')} | "
                f"Value Loss: {stats.get_average('value_loss')}"
            )
            wandb.log({
                "epoch_loss": stats.get_average('loss'),
                "epoch_ploss": stats.get_average('policy_loss'),
                "epoch_vloss": stats.get_average('value_loss'),
            })

        # Save the best model from the duels, if improved
        if best_model_state is not None:
            torch.save(best_model_state, self.cfg.BEST_MODEL_PATH)
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
    trainer.run(100)
