import torch
import torch.nn as nn
from chessbot.models import BaseChessBot
from chessbot.models import ModelRegistry


@ModelRegistry.register("simple_chessbot")
class SimpleChessBot(BaseChessBot):
    """
    A simple example model that extends the BaseChessModel for the ChessBot-Battleground library.
    This model uses a few linear layers to process the input state and produce action logits
    and a board value.
    """
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        
        # Simple mlp and some prediction heads
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, hidden_dim),  # Flatten the 8x8 board and process
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(64, self.action_dim)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Tanh()  # Output a value between -1 and 1
        )


    def forward(self, x):
        """
        Process the input state and produce action logits and board value.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, 8, 8), with integer values for pieces

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
            action_logits: shape (B, 4672)
            board_val: shape (B,1)
        """
        features = self.layers(x)  # Reshape and pass through the network
        action_logits = self.policy_head(features)
        board_val = self.value_head(features)
        return action_logits, board_val