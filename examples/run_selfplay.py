#!/usr/bin/env python3
"""
Script to run selfplay with a model and visualize the games.
"""

from chessbot.models import MODEL_REGISTRY
from chessbot.inference import selfplay

# Available model types: simple_chessbot, mixerAttn_chessbot, sgu_chessbot, swin_chessbot
MODEL_TYPE = "swin_chessbot"  # Change this to match your model architecture
WEIGHTS_PATH = "/home/kage/chess_workspace/2.13M-model.safetensors"

def main():
    # Load model with your safetensors weights
    print(f"Loading {MODEL_TYPE} with weights from {WEIGHTS_PATH}")
    model = MODEL_REGISTRY.load_with_weights(MODEL_TYPE, WEIGHTS_PATH)
    
    print("Starting selfplay with visualization...")
    # Run selfplay with visualization and MCTS search
    outcome = selfplay(
        model, 
        search=True,      # Use MCTS for stronger play
        num_sims=250,     # Number of MCTS simulations per move
        visualize=True,   # Show the game board in a window
        sample=False      # Use best move, not sampling
    )
    
    # Print the result
    if outcome == 1:
        print("White won!")
    elif outcome == -1:
        print("Black won!")
    else:
        print("Draw!")

if __name__ == "__main__":
    main()