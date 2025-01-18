import torch
import torch.nn as nn
import numpy as np

from adversarial_gym.chess_env import ChessEnv


class BaseModel(nn.Module):
    """
    Base class for all chess models in the ChessBot-Battleground library.
    Each model should accept an input tensor of shape (B, 1, 8, 8) and produce
    action logits (B, 4672) and board values (B,) as outputs.
    """

    def __init__(self):
        super().__init__()
        
        # Common loss functions for consistency's sake
        self.policy_loss = nn.CrossEntropyLoss()
        self.value_loss = nn.MSELoss()

        self.action_dim = 4672  # Typically the dimension for chess actions

    def forward(self, x):
        """
        Forward pass of the model. Should be overridden by all subclasses.

        Note that modifying inputs and outputs may require some configuration or custom code.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, 8, 8), representing the chess board state.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing action logits and board value.
                - action_logits (torch.Tensor): raw policy output of shape (B, 4672), representing the logits of each action.
                - board_val (torch.Tensor): value of the current position, in the range [-1, 1] for the current player
                    losing=-1, drawing=0, or winning=1. Shape (B,)
        """

        raise NotImplementedError("This method should be overridden by subclasses")

    def get_action(self, state, legal_moves, sample_n=1):
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
        state = (
            torch.as_tensor(state, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        policy_logits, _ = self.forward(state)

        legal_actions = [ChessEnv.move_to_action(move) for move in legal_moves]
        return self.to_action(policy_logits, legal_actions, top_n=sample_n)

    def to_action(self, action_logits, legal_actions, top_n):
        """
        Converts action logits to actual game actions, sampling from the top_n legal actions.
        This method can also be adapted by each model based on their specific needs.

        Args:
            action_logits (torch.Tensor): Logits representing the probability of each possible action.
            legal_actions (list): List of indices for legal actions.
            top_n (int): Number of top actions to consider.

        Returns:
            Tuple[int, torch.Tensor]: The chosen action and its log probability.
        """
        if len(legal_actions) < top_n:
            top_n = len(legal_actions)

        action_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)
        action_probs_np = action_probs.detach().cpu().numpy().flatten()

        # Set non legal-actions to = -inf so they aren't considered
        mask = np.ones(action_probs_np.shape, bool)
        mask[legal_actions] = False
        action_probs_np[mask] = -np.inf

        # sample from top-n policy prob indices
        top_n_indices = np.argpartition(action_probs_np, -top_n)[-top_n:]
        action = np.random.choice(top_n_indices)

        log_prob = action_probs.flatten()[action]
        return action, log_prob
