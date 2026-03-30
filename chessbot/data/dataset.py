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

    _counts = None  # class-level cache

    @classmethod
    def get_counts(cls):
        if cls._counts is None:
            try:
                from huggingface_hub import hf_hub_download
                import json
                path = hf_hub_download(repo_id="KeithG33/ChessBot-Dataset", filename="count.txt", repo_type="dataset")
                with open(path) as f:
                    cls._counts = json.loads(f.readline())
            except Exception:
                cls._counts = {}
        return cls._counts

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
        self.total = self.get_counts().get(split)
        print(f"Total samples in {split} set: {self.total}")
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
            state = example["state"]                 # torch.int8 tensor shape [8,8]
            action = example["action"]               # torch.int16 scalar tensor
            best_actions = example["best_actions"]   # torch.int16 tensor of best action indices, empty if absent
            legal_actions = example["legal_actions"] # torch.int16 tensor of legal action indices
            result = example["result"]               # float32 scalar tensor

            action_vec = torch.zeros(4672, dtype=torch.float32)
            action_vec[legal_actions.long()] = 0.1

            if len(best_actions) > 0:
                action_vec[best_actions.long()] = 1.0
                if action.item() not in best_actions:
                    action_vec[action.long()] = 0.9
            else:
                action_vec[action.long()] = 1.0

            yield state, action_vec, result


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

    @staticmethod
    def _parse_best_moves(comment, board):
        """Parse UCI best moves from a comment like '0.52 e2e4' or '1.0 e1e2 h7h8q a1a8'.
        Returns list of action indices. Supports multiple moves (tablebase-generated data)."""
        if not comment:
            return []
        parts = comment.split()
        if len(parts) < 2:
            return []
        # First token must be a q-value float; remaining tokens are UCI moves
        try:
            float(parts[0])
        except ValueError:
            return []
        best_actions = []
        for token in parts[1:]:
            try:
                move = chess.Move.from_uci(token)
                if move in board.legal_moves:
                    best_actions.append(int(ChessEnv.move_to_action(move)))
            except Exception:
                break
        return best_actions

    def generate_pgn_data(self, pgn_file):
        data = []
        with open(pgn_file) as pgn:
            while (game := chess.pgn.read_game(pgn)) is not None:
                state_maps, actions, results, best_actions_list, source = self.get_game_data(game)
                game_data = list(zip(state_maps, actions, results, best_actions_list, [source] * len(actions)))
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
        best_actions_list = []

        board = game.board()
        result = game.headers['Result']
        result = result_to_number(result)
        source = "tablebase" if game.headers.get("Event") == "Tablebase" else "game"
        for node in game.mainline():
            move = node.move
            canon_state = self.get_canonical_state(board, board.turn)
            action = ChessEnv.move_to_action(move)
            best_actions = self._parse_best_moves(node.comment, board)

            states.append(canon_state)
            actions.append(action)
            best_actions_list.append(best_actions)
            results.append(
                result * -1 if board.turn == 0 else result
            )  # flip value to match canonical state
            board.push(move)

        if game.errors:
            print(game.errors, game.headers)

        return states, actions, results, best_actions_list, source

    def get_canonical_state(self, board, turn):
        state = ChessEnv.get_piece_configuration(board)
        state = state if turn else -state
        return state

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action, result, best_actions, source = self.data[idx]
        if source == "tablebase":
            # All best_actions are equally optimal tablebase moves — all 1.0
            # The played move IS one of the best moves so it also gets 1.0
            probs = {ba: 1.0 for ba in best_actions}
        elif len(best_actions) == 1:
            # Single best move from lc0/engine annotation: best move is 1.0, played move is 0.9
            probs = {best_actions[0]: 1.0, action: 0.9}
        else:
            probs = {action: 1.0}
        action_probs = torch.tensor(create_sparse_vector(probs), dtype=torch.float32)
        return state, action_probs, result


