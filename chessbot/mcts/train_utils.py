import random
import os

import gym
import chess

from multiprocessing.managers import BaseManager, NamespaceProxy

import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
from torch.multiprocessing import Pool, Process, Lock, Event

from timm.optim import create_optimizer_v2, list_optimizers

from chessbot.train.utils import MetricsTracker
from chessbot.data import ChessDataset
from chessbot.mcts.search import MonteCarloTreeSearch
from chessbot.inference import play_game
from chessbot.models import align_state_dict, MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Model loading and saving utilities
# ---------------------------------------------------------------------------

def torch_safesave(state_dict, path, file_lock):
    with file_lock:
        torch.save(state_dict, path)

def torch_safeload(path, file_lock):
    with file_lock:
        model_state = align_state_dict(torch.load(path, weights_only=True))
    return model_state
    
def load_model(cfg, state_dict, mode='eval'):
    model = MODEL_REGISTRY.load_model(cfg.MODEL_CLASS, *cfg.MODEL_ARGS, **cfg.MODEL_KWARGS)
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval() if mode == 'eval' else model.train() if mode == 'train' else ValueError("Invalid mode")
    return model

# ---------------------------------------------------------------------------
# Self-play and training
# ---------------------------------------------------------------------------

def run_games_continuously(
    cfg,
    shutdown_event,
    global_counter,
    file_lock,
    replay_buffer,
):
    """Continuously run self-play games in parallel. Saves data to shared replay buffer."""

    with Pool(processes=cfg.SELFPLAY_PARALLEL) as pool:
        while not shutdown_event.is_set():
            # Load current best model
            model_state = torch_safeload(cfg.MODEL_BEST_PATH, file_lock)

            # Start games in parallel without blocking. Each game runs in a separate process.
            async_results = [
                pool.apply_async(run_selfplay_game, args=(cfg, model_state, global_counter, cfg.SELFPLAY_SIMS))
                for _ in range(cfg.SELFPLAY_PARALLEL)
            ]

            # Iterate over the results as they complete
            for async_result in async_results:
                g_states, g_actions, reward = async_result.get()
                file_lock.acquire()
                try:
                    replay_buffer.update(g_states, g_actions, reward)
                finally:
                    file_lock.release()

def run_selfplay_game(cfg, model_state, global_counter, num_simulations=650):
    env = gym.make("Chess-v0")
    observation, info = env.reset()

    model = load_model(cfg, model_state, mode='train')
    tree = MonteCarloTreeSearch(env, model)

    terminal = False
    game_actions = []
    game_states = []
    while not terminal:
        state = env.get_string_representation()
        best_action, action_probs = tree.search(state, observation, num_simulations)
        game_actions.append(action_probs)
        game_states.append(observation[0])
        observation, reward, terminal, trunc, info = env.step(best_action)
        
    global_counter.increment()
    return game_states, game_actions, reward

# ---------------------------------------------------------------------------
# Dueling
# ---------------------------------------------------------------------------

def convert_outcome(outcome, perspective):
    if outcome == 0:
        return 0.5
    elif perspective == chess.BLACK and outcome == -1:
        return 1
    elif perspective == chess.WHITE and outcome == 1:
        return 1
    return 0


def play_match_game(args):
    """Set up and play a match game between new and old model and return the score."""
    cfg, new_model_state, old_model_state, perspective, num_sims = args
    new_model = load_model(cfg, new_model_state, mode='eval')
    old_model = load_model(cfg, old_model_state, mode='eval')

    if perspective == chess.WHITE: # new model is white (from state dict)
        outcome = play_game(new_model, old_model, search=True, num_sims=num_sims)
    elif perspective == chess.BLACK: # new model is black (from state dict)
        outcome = play_game(old_model, new_model, search=True, num_sims=num_sims)
    score = convert_outcome(outcome, perspective)
    
    return score


def run_play_match(cfg, new_model_path, old_model_path, num_rounds, file_lock, num_sims=100, num_processes=2):
    """Play matches between the new and old model and return the score using parallel processes."""
    
    scores = []
    wins, losses, draws = 0, 0, 0

    new_model_state = torch_safeload(new_model_path, file_lock)
    old_model_state = torch_safeload(old_model_path, file_lock)
    
    # NOTE: might need this
    # new_model_state = {k: v.cpu() for k, v in new_model_state.items()} # can't share cuda tensors
    # old_model_state = {k: v.cpu() for k, v in old_model_state.items()} # can't share cuda tensors
    
    # Args for white and black
    args_list = [((cfg, new_model_state, old_model_state, chess.WHITE, num_sims)) for _ in range(num_rounds)]
    args_list += [((cfg, new_model_state, old_model_state, chess.BLACK, num_sims)) for _ in range(num_rounds)]

    with Pool(processes=num_processes) as pool:
        scores = pool.map(play_match_game, args_list)

    for score in scores:
        if   score == 0:    losses += 1
        elif score == 0.5:  draws  += 1
        elif score == 1:    wins   += 1

    score = sum(scores)

    return {"score": score, "wins": wins, "draws": draws, "losses": losses}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_dataset(replay_buffer, cfg):
    """
    Builds datasets by sampling expert games and combining with selfplay data.
    Prioritizes the use of expert data and fills the remainder with selfplay data.

    Args:
    - mp_manager: Multiprocessing manager with selfplay data.
    - cfg: Configuration object containing dataset paths and sizes.

    Returns:
    - A combined dataset ready for training.
    """
    # Initialize selfplay dataset
    train_dataset = ChessReplayDataset(replay_buffer)
    selfplay_size = len(train_dataset)

    if selfplay_size > cfg.SELFPLAY_BUFFER_SIZE:
        indices = random.sample(range(selfplay_size), cfg.SELFPLAY_BUFFER_SIZE)
        train_dataset = Subset(train_dataset, indices)
        return train_dataset
    
    if cfg.TRAIN_WITH_EXPERT:
        # Fill remaining dataset with expert data, with optional minimum ratio 
        expert_files = [
            pgn.path for pgn in os.scandir(cfg.TRAIN_PGN_DIR) if pgn.name.endswith(".pgn")
        ]
        sampled_expert_files = random.sample(expert_files, cfg.TRAIN_EXPERT_SIZE)
        expert_dataset = ChessDataset(
            sampled_expert_files, load_parallel=True, num_threads=10
        )

        expert_size = cfg.TRAIN_DATASET_SIZE - selfplay_size
        expert_size = min(cfg.TRAIN_EXPERT_RATIO * cfg.TRAIN_DATASET_SIZE, expert_size)
        indices = random.sample(range(len(expert_dataset)), expert_size)
        expert_subset = Subset(expert_dataset, indices)
        train_dataset = ConcatDataset([expert_subset, train_dataset])

    return train_dataset


def build_optimizer(model, cfg):
    optimizer_str = cfg.TRAIN_OPTIMIZER.lower()
    assert (optimizer_str in list_optimizers()) or None, f"Optimizer {optimizer_str} not valid - must be one of: {list_optimizers()}"
    
    if optimizer_str == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=cfg.TRAIN_LR
        )
    elif optimizer_str == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=cfg.TRAIN_LR, 
            momentum=0.9
        )
    elif optimizer_str is not None:
        optimizer = create_optimizer_v2(
            model.parameters(),
            opt=optimizer_str,
            lr=cfg.TRAIN_LR
        )
   
    scheduler = None

    if cfg.TRAIN_SCHEDULER == "linear":
        min_scale = cfg.TRAIN_MIN_LR / cfg.TRAIN_LR
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_scale,
            total_iters=cfg.TRAIN_SCHEDULER_ITERS,
        )
    elif cfg.TRAIN_SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            cfg.TRAIN_SCHEDULER_ITERS,
            cfg.TRAIN_MIN_LR,
        )
    
    if scheduler is not None and cfg.TRAIN_WARMUP_ITERS > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=cfg.TRAIN_WARMUP_LR / cfg.TRAIN_LR,
            end_factor=1.0,
            total_iters=cfg.TRAIN_WARMUP_ITERS,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[cfg.TRAIN_WARMUP_ITERS]
        )
    return optimizer, scheduler


def run_training_epoch(shared_replay_buffer, cfg):
    # Load current model (don't need safeload here)
    model_state = align_state_dict(torch.load(cfg.MODEL_CURR_PATH, weights_only=True))
    model = load_model(cfg, model_state, mode='train')

    stats = MetricsTracker()
    stats.add("loss", "policy_loss", "value_loss")

    train_dataset = build_dataset(shared_replay_buffer, cfg)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN_BATCH_SIZE, shuffle=True)

    optimizer, scheduler = build_optimizer(model, cfg)

    # If an optimizer checkpoint exists, load it.
    if os.path.exists(cfg.TRAIN_STATE_PATH):
        ckpt = torch.load(cfg.TRAIN_STATE_PATH, weights_only=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

    for i, (states_batch, actions_batch, values_batch) in enumerate(train_loader):
        states_batch = states_batch.to(model.device, dtype=torch.float32).unsqueeze(1)
        actions_batch = actions_batch.to(model.device,)
        values_batch = values_batch.to(model.device, dtype=torch.float32)

        # AMP with grad clipping
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            policy_output, value_output = model(states_batch)
            policy_loss = model.policy_loss(policy_output.squeeze(), actions_batch)
            value_loss = model.value_loss(value_output.squeeze(), values_batch)
            loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Record the losses
        stats.update(
            {
                "loss": loss.item(),
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
            }
        )

    # Epoch done - save model for match evaluation, optimizer
    torch.save(model.state_dict(), cfg.MODEL_CURR_PATH)
    
    # Save optimizer and scheduler states for the next epoch.
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict()
    }, cfg.TRAIN_STATE_PATH)
    
    return stats


# ---------------------------------------------------------------------------
# Replay Buffer and Dataset
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """Replay buffer to store past experiences for training policy/value network"""

    def __init__(self, capacity=None):
        self.action_probs = []
        self.states = []
        self.values = []

        self.capacity = capacity
        self.curr_length = 0
        self.position = 0

    def get_state(self):
        return {
            'actions': list(self.action_probs),
            'states': list(self.states),
            'values': list(self.values),
            'capacity': self.capacity,
            'curr_length': self.curr_length,
            'position': self.position,
        }

    def from_dict(self, buffer_state_dict):
        for key, value in buffer_state_dict.items():
            setattr(self, key, value)

    def push(self, state, action, value):
        if len(self.action_probs) < self.capacity:
            self.states.append(None)
            self.action_probs.append(None)
            self.values.append(None)

        self.states[self.position] = state
        self.action_probs[self.position] = action
        self.values[self.position] = value

        self.curr_length = len(self.states)
        self.position = (self.position + 1) % self.capacity

    def update(self, states, actions, winner):
        # Create value targets based on who won
        if winner == 1:
            values = [(-1) ** (i) for i in range(len(states))]
        elif winner == -1:
            values = [(-1) ** (i + 1) for i in range(len(states))]
        else:
            values = [0] * len(states)

        for state, action, value in zip(states, actions, values):
            self.push(state, action, value)

    def clear(self):
        self.action_probs = []
        self.states = []
        self.values = []
        self.curr_length = 0
        self.position = 0


class ChessReplayDataset(Dataset):
    """Dataset class for replay buffer data. Uses shared replay buffer object"""

    def __init__(self, replay_buffer_proxy):
        # Initialize a dataset with replay buffer data
        self.replay_buffer = ReplayBuffer()
        self.replay_buffer.from_dict(replay_buffer_proxy.get_state())

    def __len__(self):
        return self.replay_buffer.curr_length

    def create_sparse_vector(self, action_probs):
        sparse_vector = [0.0] * 4672
        for action, prob in action_probs.items():
            sparse_vector[action] = prob
        return sparse_vector

    def get_state(self):
        return {'replay_buffer': self.replay_buffer}

    def __getitem__(self, idx):
        state = self.replay_buffer.states[idx]
        value = self.replay_buffer.values[idx]
        action_probs = self.replay_buffer.actions[idx]
        action_probs = self.create_sparse_vector(action_probs)
        action_probs = torch.tensor(action_probs, dtype=torch.float32)
        return state, action_probs, value


""" 
Helper classes and setup for training with multiprocessing. 

These classes enable the replay buffer to be shared and used across multiple processes.
"""

# ---------------------------------------------------------------------------
# Game Counter and Proxy
# ---------------------------------------------------------------------------

class GameCounter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

class GameCounterProxy(NamespaceProxy):
    # We need to expose the same __dunder__ methods as NamespaceProxy,
    # in addition to the b method.
    _exposed_ = ('__getattribute__', '__setattr__', '__delattr__', 'increment')

    def increment(self):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod('increment')


# ---------------------------------------------------------------------------
# Replay Buffer Manager Setup
# ---------------------------------------------------------------------------

class ReplayBufferManager(BaseManager):
    pass

# Expose ReplayBuffer methods to the manager
ReplayBufferManager.register(
    typeid='ReplayBuffer',
    callable=ReplayBuffer,
    exposed=['from_dict', 'set_state', 'get_state', 'update', 'clear'],
)
# Register functionality for event handling, file-lock handling, game counting
ReplayBufferManager.register('Event', Event)
ReplayBufferManager.register('Lock', Lock)
ReplayBufferManager.register('GameCounter', GameCounter, GameCounterProxy)

# ---------------------------------------------------------------------------
# Self-Play MultiProcessing Manager
# ---------------------------------------------------------------------------
class SelfPlayMultiProcManager:
    """Handles the various helper objects and functionality for running selfplay and training"""

    def __init__(self, replay_buffer_capacity):
        self.manager = ReplayBufferManager()
        self.manager.start()
        self.shared_replay_buffer = self.manager.ReplayBuffer(capacity=replay_buffer_capacity)
        self.shutdown_event = self.manager.Event()
        self.buffer_lock = self.manager.Lock()
        self.global_game_counter = self.manager.GameCounter()
        self.file_lock = Lock()
        self.process = None

    def start_self_play_process(
        self, cfg, shutdown_event, global_counter, file_lock, shared_replay_buffer
    ):
        self.process = Process(
            target=run_games_continuously,
            args=(cfg, shutdown_event, global_counter, file_lock, shared_replay_buffer),
        )
        self.process.start()

    def join_process(self):
        if self.process:
            self.process.join()