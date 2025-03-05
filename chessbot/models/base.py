from typing import List

import numpy as np
import torch
import torch.nn as nn

from adversarial_gym.chess_env import ChessEnv

import chess

from chessbot.models import align_state_dict


def align_state_dict(state_dict, prefix="_orig_mod.", new_prefix=""):
    """ Fix keys of a compiled model's state_dict, which have _orig_mod. prefixed to start """
    new_state_dict = {}

    # replace the specified prefix to each key
    for key, value in state_dict.items():
        new_key = key.replace(prefix, new_prefix)
        new_state_dict[new_key] = value
    return new_state_dict


class BaseChessBot(nn.Module):
    """
    Base class for all chess models in the ChessBot-Battleground library.
    Each model should accept an input tensor of shape (B, 1, 8, 8) and produce
    action logits (B, 4672) and board values (B,) as outputs.
    """

    def __init__(self):
        super().__init__()

        self.action_dim = 4672

    def load_weights(self, path) -> None:
        """Load weights from a file. Fix added prefix from compilation if needed."""
        weights = align_state_dict(torch.load(path))
        self.load_state_dict(weights)

    def get_current_device(self):
        """Get the current device of the model"""
        return next(self.parameters()).device

    def forward(self, x, *args, **kwargs):
        """
        Forward pass of the model. Should be overridden by all subclasses.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, 8, 8), representing the chess board state.

        Returns:
            - action_logits (torch.Tensor): raw policy output of shape (B, 4672), representing the logits of each action.
            - board_val (torch.Tensor): value output of the current position with shape (B,), in the range [-1, 1] for current player
                losing=-1, drawing=0, or winning=1.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def __call__(self, *args, **kwargs):
        """Override the __call__ method to automatically add numpy to torch conversion

        Needed because the PyTorch dataset returns (B, 1, 8, 8) tensors but the gym environment
        returns (8, 8) numpy arrays.

        Expects the input tensor is first arg
        """

        if isinstance(args[0], np.ndarray):
            input = torch.as_tensor(args[0], dtype=torch.float32).reshape(
                1, 1, *args[0].shape
            )
            args = (input, *args[1:])

        return super().__call__(*args, **kwargs)

    def get_action(self, state: np.ndarray, legal_moves: List[chess.Move], sample=False):
        """
        Given the state and legal moves, returns the selected action and its log probability.
        Used when interacting with the ChessEnv

        Args:
            state (numpy.ndarray): The current state of the chess board.
            legal_moves (list): List of legal moves in the current state.
            sample_n (int): Number of top actions to sample from.

        Returns:
            Tuple[int, float]: The chosen action and its log probability.
        """
        # Put input on current device
        device = self.get_current_device()
        state = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(
            1, 1, 8, 8
        )

        policy_logits, _ = self.forward(state)

        legal_actions = [ChessEnv.move_to_action(move) for move in legal_moves]
        return self.to_action(policy_logits, legal_actions, sample=sample)

    def to_action(
        self,
        action_logits: torch.Tensor,
        legal_actions: list[int],
        sample: bool = False,
    ) -> tuple[int, torch.Tensor]:
        """
        Converts action logits to a game action and log-prob. Filters out illegal moves and then
        samples or selects highest from policy distribution.

        Args:
            action_logits (torch.Tensor): Raw logits of shape (1, 4672) representing all possible actions.
            legal_actions (list[int]): List of indices corresponding to legal moves.
            sample (bool): If True, sample from the policy distribution; if False, choose the top action.

        Returns:
            tuple[int, torch.Tensor]: The chosen action index and its log probability.
        """
        logits = action_logits.flatten()
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[legal_actions] = False
        masked_logits = logits.clone()
        masked_logits[mask] = float('-inf')

        if sample:
            dist = torch.distributions.Categorical(logits=masked_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        else:
            log_probs = torch.nn.functional.log_softmax(masked_logits, dim=-1)
            action = torch.argmax(masked_logits)
            log_prob = log_probs[action]

        return int(action.item()), log_prob