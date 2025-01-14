import gym
import adversarial_gym
import chess

from chessbot.models.base import BaseModel
from chessbot.mcts import MonteCarloTreeSearch


def play_game(white: BaseModel, black: BaseModel, perspective=None, num_sims=100, visualize=False):
    """
    Plays a game and returns 1 if chosen perspective has won, else 0.

    Perspective is either Chess.WHITE (1) or Chess.BLACK (0).
    """
    step = 0
    done = False
    env = gym.make("Chess-v0", render_mode='human') if visualize else gym.make("Chess-v0")
    obs, info = env.reset()

    white_tree = MonteCarloTreeSearch(env, white)
    black_tree = MonteCarloTreeSearch(env, black)

    while not done:
        state = env.get_string_representation()
        if step % 2 == 0:
            _, best_action = white_tree.search(
                state, obs, simulations_number=num_sims
            )
        else:
            _, best_action = black_tree.search(
                state, obs, simulations_number=num_sims
            )

        obs, reward, done, _, _ = env.step(best_action)
        step += 1

    # Return points for win/loss/draw
    if reward == 0:
        score = 0.5
    elif perspective == chess.BLACK and reward == -1:
        score = 1
    elif perspective == chess.WHITE and reward == 1:
        score = 1
    else:
        score = 0

    return score
