from typing import Any, Optional
import gym
import adversarial_gym
import chess

from chessbot.models.base import BaseChessBot
from chessbot.mcts import MonteCarloTreeSearch


def score_function(outcome, perspective) -> float | int:
    """
    Return 1 of white won, -1 if white lost, 0 for draw --> perspective=chess.WHITE
    Return 1 of black won, -1 if black lost, 0 for draw --> perspective=chess.BLACK
    """
    if outcome == 0:
        return 0.5
    if (
        perspective == chess.WHITE
        and outcome == 1
        or perspective == chess.BLACK
        and outcome == -1
    ):
        return 1

    return 0


def duel(
    player1: BaseChessBot,
    player2: BaseChessBot,
    best_of=7,
    search=False,
    num_sims=250,
    visualize=False,
    sample=False,
) -> tuple[int, int]:
    """
    Conducts a duel (best-of series) between two chess models and returns their respective scores.

    The final scores are computed with each win contributing +1, each loss -1, and draws contributing 0.

    Args:
        model1 (BaseChessModel): The first chess model.
        model2 (BaseChessModel): The second chess model.
        best_of (int, optional): The number of games to be played in the duel. Defaults to 3.
        search (bool, optional): If True, use MCTS for move selection in the games. Defaults to True.
        visualize (bool, optional): If True, render the games visually. Defaults to False.
        sample (bool, optional): If True, sample moves from the output distribution. Not relevant if search=True

    Returns:
        tuple[int, int]: A tuple containing the cumulative scores for model1 and model2 respectively.
    """
    player1_score = 0
    player2_score = 0

    win_condition = (best_of // 2) + 1

    for i in range(best_of):
        if i % 2 == 0:
            outcome = play_game(
                player1,
                player2,
                search=search,
                num_sims=num_sims,
                visualize=visualize,
                sample=sample,
            )
            player1_score += score_function(outcome, chess.WHITE)
            player2_score += score_function(outcome, chess.BLACK)
        else:
            outcome = play_game(
                player2,
                player1,
                search=search,
                num_sims=num_sims,
                visualize=visualize,
                sample=sample,
            )
            player1_score += score_function(outcome, chess.BLACK)
            player2_score += score_function(outcome, chess.WHITE)

        if any(score >= win_condition for score in (player1_score, player2_score)):
            break

    return player1_score, player2_score


def play_game(
    white: BaseChessBot,
    black: BaseChessBot,
    search=False,
    num_sims=250,
    visualize=False,
    sample=False,
) -> int:
    """
    Plays a game and returns 1 if white has won, -1 if black has won, and 0 for a draw.
    """
    step = 0
    done = False
    env = (
        gym.make("Chess-v0", render_mode='human') if visualize else gym.make("Chess-v0")
    )
    obs, info = env.reset()

    if search:
        white = MonteCarloTreeSearch(env, white)
        black = MonteCarloTreeSearch(env, black)

    while not done:
        state = env.get_string_representation()
        legal_moves = env.board.legal_moves

        if step % 2 == 0:
            best_action, _ = (
                white.search(state, obs, num_simulations=num_sims)
                if search
                else white.get_action(obs[0], legal_moves, sample=sample)
            )
        else:
            best_action, _ = (
                black.search(state, obs, num_simulations=num_sims)
                if search
                else black.get_action(obs[0], legal_moves, sample=sample)
            )

        obs, reward, done, trunc, _ = env.step(best_action)
        step += 1

    return reward


def selfplay(
    model: BaseChessBot, search=False, num_sims=250, visualize=False, sample=False
) -> int:
    """
    Conducts a self-play game with the given model.

    The game is played until completion, with the option to use Monte Carlo Tree Search (MCTS)
    for decision-making, a specified number of simulations, and optional visualization. The outcome
    is returned as 1 for a white win, -1 for a black win, or 0 for a draw.

    Args:
        model (BaseChessModel): The chess model used for both sides.
        search (bool, optional): If True, utilize MCTS for move selection. Defaults to True.
        num_sims (int, optional): The number of simulations for MCTS. Defaults to 250.
        visualize (bool, optional): If True, render the game visually. Defaults to False.
        sample (bool, optional): If True, sample moves from the output distribution. Not relevant if search=True

    Returns:
        int: The game outcome (1 for white win, -1 for black win, 0 for draw).
    """
    return play_game(
        model,
        model,
        search=search,
        num_sims=num_sims,
        visualize=visualize,
        sample=sample,
    )
