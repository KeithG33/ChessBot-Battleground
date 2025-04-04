from pathlib import Path
from typing import List
import chess
import chess.pgn

import torch
from torch.multiprocessing import Pool, set_start_method
from torch.utils.data import Dataset

from adversarial_gym.chess_env import ChessEnv


def create_sparse_vector(action_probs):
    # Initialize a list of zeros for all possible actions
    sparse_vector = [0.0] * 4672

    for action, prob in action_probs.items():
        sparse_vector[action] = prob
    return sparse_vector


def result_to_number(result):
    if result == "1-0":
        return 1
    if result == "0-1":
        return -1
    return 0


class ChessDataset(Dataset):
    """
    The PyTorch chess dataset to load and process game data from PGN files.

    For each board position, the dataset extracts:
      - The board state.
      - The move taken (action), represented as a one-hot vector of shape (4672,)
      - The game result: 1 for win, -1 for loss, 0 for draw

    The dataset supports both sequential and parallel processing of PGN files
    with the `num_processes` argument.
    """

    def __init__(self, pgn_files, num_processes = 0):
        super().__init__()
        self.load_pgn_files(pgn_files)
        self.data = (
            self.generate_data()
            if num_processes <= 1
            else self.generate_data_parallel(num_processes)
        )

    def load_pgn_files(self, pgn_files):
        """Handle loading pgn files from a directory, file, or list of files"""
        if isinstance(pgn_files, str):
            if Path(pgn_files).is_dir():
                self.pgn_files = list(Path(pgn_files).rglob("*.pgn"))
            elif Path(pgn_files).is_file():
                self.pgn_files = [pgn_files]
            else:
                raise ValueError("Invalid pgn_files argument - string not a file or directory"
                )
        elif isinstance(pgn_files, list):
            self.pgn_files = pgn_files
        else:
            raise ValueError("Invalid pgn_files argument")

    def generate_pgn_data(self, pgn_file):
        data = []
        with open(pgn_file) as pgn:
            while (game := chess.pgn.read_game(pgn)) is not None:
                state_maps, actions, results = self.get_game_data(game)
                game_data = list(zip(state_maps, actions, results))
                data.extend(game_data)
        return data
    
    def generate_data(self):
        """Return a list of tuples of (state, action, result)"""
        data = []
        for pgn_file in self.pgn_files:
            data.extend(self.generate_pgn_data(pgn_file))
        return data

    def generate_data_parallel(self, num_processes):
        try:
            set_start_method('spawn', force=True)
        except RuntimeError as err:
            print(f"Warning: {err}")

        with Pool(num_processes) as pool:
            results = pool.map(self.generate_pgn_data, self.pgn_files)

        return [item for sublist in results for item in sublist]

    def get_game_data(self, game):
        """
        Generate a game's training data, where X = board_state and Y1 = action taken, Y2 = result.
        A move is given by [from_square, to_square, promotion, drop]
        """
        states = []
        actions = []
        results = []

        board = game.board()
        result = game.headers['Result']
        result = result_to_number(result)
        for move in game.mainline_moves():
            canon_state = self.get_canonical_state(board, board.turn)
            action = ChessEnv.move_to_action(move)

            states.append(canon_state)
            actions.append(action)
            results.append(
                result * -1 if board.turn == 0 else result
            )  # flip value to match canonical state
            board.push(move)

        if game.errors:
            print(game.errors, game.headers)

        return states, actions, results

    def get_canonical_state(self, board, turn):
        state = ChessEnv.get_piece_configuration(board)
        state = state if turn else -state
        return state

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action, result = self.data[idx]
        action_probs = create_sparse_vector({action: 1.0})
        action_probs = torch.tensor(action_probs, dtype=torch.float32)
        return state, action_probs, result
