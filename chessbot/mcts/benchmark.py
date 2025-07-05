import time

import gym
import torch

from .search import MonteCarloTreeSearch
from .fast_search import FastMonteCarloTreeSearch
from chessbot.models import MODEL_REGISTRY


def benchmark_mcts(num_simulations=100):
    """Benchmark old and optimized MCTS implementations."""
    env = gym.make("Chess-v0")
    obs, _ = env.reset()
    model = MODEL_REGISTRY.load_model("simple_chessbot")
    model.eval()

    state = env.get_string_representation()

    tree1 = MonteCarloTreeSearch(env, model)
    start = time.time()
    tree1.search(state, obs, num_simulations)
    old_time = time.time() - start

    tree2 = FastMonteCarloTreeSearch(env, model)
    start = time.time()
    tree2.search(state, obs, num_simulations)
    new_time = time.time() - start

    return {"standard": old_time, "optimized": new_time}
