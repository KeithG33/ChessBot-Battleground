import torch
import torch.nn as nn
import numpy as np

from adversarial_gym.chess_env import ChessEnv


class BaseChessModel(nn.Module):
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

        self.action_dim = 4672

        self.validate_flag = True

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """
        Automatically injects validation at the end of the derived class `__init__()`.
        """
        original_init = cls.__init__

        def wrapped_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)  # Call the original __init__
            
            # Validate the model after initialization if self.validate_flag is True
            if getattr(self, "validate", True):
                self.validate_model()

        cls.__init__ = wrapped_init  # Replace the subclass's __init__ with the wrapped version

        super().__init_subclass__(**kwargs)  # Call super to ensure proper subclass behavior

    def validate_model(self):
        """ Quickly validate model meets expected input/output shapes """
        test_batch = 2
        policy_shape = (test_batch, 4672)
        value_shape1 = (test_batch,1)
        value_shape2 = (test_batch,)

        x = torch.randn(2, 1, 8, 8).cpu()
        self = self.cpu()
        output1, output2 = self.forward(x)

        if output1.shape != policy_shape:
            raise ValueError(f"Model Validation Fail - expected policy output to have shape (B, 4672), but got {output1.shape}")

        if output2.shape not in [value_shape1, value_shape2]:
            raise ValueError(f"Model Validation Fail - Expected value output to have shape (B,), but got {output2.shape}")

        print("✅ Model validated successfully!")

    def get_current_device(self):
        """ Get the current device of the model """
        return next(self.parameters()).device

    def forward(self, x):
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
        """ Override the __call__ method to automatically add numpy to torch conversion
        
        Needed because the PyTorch dataset returns (B, 1, 8, 8) tensors but the gym environment
        returns (8, 8) numpy arrays.

        Expects the input x tensor is first arg
        """

        if isinstance(args[0], np.ndarray):
            input = torch.as_tensor(args[0], dtype=torch.float32).reshape(1, 1, *args[0].shape)
            args = (input, *args[1:])

        return super().__call__(*args, **kwargs)
    
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
        # Put input on current device
        device = self.get_current_device()
        state = (
            torch.as_tensor(state, dtype=torch.float32, device=device)
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


# # test model
# model = BaseChessModel()
# model.validate_model()