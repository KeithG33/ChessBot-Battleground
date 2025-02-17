import gym
import adversarial_gym
import chess

from chessbot.models.base import BaseChessModel
from chessbot.mcts import MonteCarloTreeSearch


def score_function(outcome, perspective):
    """ Return score based on outcome (outcome of game) and perspective (color of player). """
    if outcome == 0:
        return 0.5
    
    if (perspective == chess.WHITE and outcome == 1
        or perspective == chess.BLACK and outcome == -1
    ):
        return 1

    return 0


def play_match(player1: BaseChessModel, player2: BaseChessModel, best_of=7, search=False, num_sims=250, visualize=False):
    """ Play a match between two models and return the score of each model. """
    player1_score = 0
    player2_score = 0

    for i in range(best_of):
        if i % 2 == 0:
            outcome = play_game(player1, player2, search=search, num_sims=num_sims, visualize=visualize)
            player1_score += score_function(outcome, chess.WHITE)
            player2_score += score_function(outcome, chess.BLACK)
        else:
            outcome = play_game(player2, player1, search=search, num_sims=num_sims, visualize=visualize)
            player1_score += score_function(outcome, chess.BLACK)
            player2_score += score_function(outcome, chess.WHITE)

    return player1_score, player2_score


def play_game(white: BaseChessModel, black: BaseChessModel, search=False, num_sims=250, visualize=False):
    """
    Plays a game and returns 1 if white has won, -1 if black has won, and 0 for a draw.
    """
    step = 0
    done = False
    env = gym.make("Chess-v0", render_mode='human') if visualize else gym.make("Chess-v0")
    obs, info = env.reset()

    if search:
        white = MonteCarloTreeSearch(env, white)
        black = MonteCarloTreeSearch(env, black)

    while not done:
        state = env.get_string_representation()
        legal_moves = env.board.legal_moves

        if step % 2 == 0:
            best_action, _ = white.search(
                state, obs, simulations_number=num_sims
            ) if search else white.get_action(obs[0], legal_moves)
        else:
            best_action, _ = black.search(
                state, obs, simulations_number=num_sims
            ) if search else black.get_action(obs[0], legal_moves)

        obs, reward, done, _, _ = env.step(best_action)
        step += 1
    
    return reward


def selfplay(model: BaseChessModel, visualize=False):
    """
    Run selfplay game with a given model.
    """
    return play_game(model, model, visualize=visualize)

     