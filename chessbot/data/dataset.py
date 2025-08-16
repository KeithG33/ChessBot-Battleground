from pathlib import Path
import random
from typing import List, Optional
import chess
import chess.pgn

import torch
from torch.multiprocessing import Pool, set_start_method
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset
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


class HFChessDataset(IterableDataset):
    """ A wrapper for streaming the ChessBot dataset on Hugging Face """
    def __init__(
        self,
        split: str = "train",
        streaming: bool = True,
        shuffle_buffer: Optional[int] = 10_000,
        num_test_samples: Optional[int] = 10_000_000, # approximately 1/3 of test set
        take_samples: Optional[int] = None,  # Number of samples to take from dataset
    ):
        super().__init__()
        self.split = split
        self.ds = load_dataset(
            "KeithG33/ChessBot-Dataset",
            data_files={
                "train": "train/*.pgn.zst",
                "test":  "test/*.pgn.zst",
            },
            split=split,
            streaming=streaming,
            trust_remote_code=True,
        ).with_format("torch")
        
        self.shuffle_buffer = shuffle_buffer
        self.num_test_samples = num_test_samples
        self.take_samples = take_samples

    def __iter__(self):
        self.ds = self.ds.shuffle(buffer_size=self.shuffle_buffer, seed=random.randint(0, 2**32-1))

        if self.split == "test" and self.num_test_samples is not None:
            self.ds = self.ds.take(self.num_test_samples)
        elif self.split == "train" and self.take_samples is not None:
            self.ds = self.ds.take(self.take_samples)
            
        for example in self.ds:
            state = example["state"]            # torch.int8 tensor shape [8,8]
            action = example["action"]          # torch.int16 scalar tensor
            result = example["result"]          # torch.int8 scalar tensor
            
            action = torch.nn.functional.one_hot(action.long(), num_classes=4672).to(torch.float32)
            yield state, action, result


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


