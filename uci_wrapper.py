#!/usr/bin/env python3
"""
UCI wrapper for ChessBot-Battleground neural network models.

Makes a trained model speak the Universal Chess Interface (UCI) protocol,
enabling it to work with any UCI-compatible tool: cutechess-cli, Arena, etc.

Usage:
    python uci_wrapper.py --model sgu_chessbot --weights path/to/model.safetensors
    python uci_wrapper.py --model sgu_chessbot --weights path/to/model.safetensors --search --num-sims 100
    python uci_wrapper.py --model-path ./models/my_custom_model --model my_model --weights path/to/weights.safetensors

Pipe to test:
    echo -e "uci\\nucinewgame\\nposition startpos\\ngo movetime 1000\\nquit" | python uci_wrapper.py --model sgu_chessbot --weights path/to/model.safetensors
"""

import argparse
import sys
import os

import chess
import numpy as np
import torch

from chessbot.models.registry import ModelRegistry, auto_register_models
from chessbot.mcts.search import _action_to_move


ENGINE_NAME = "ChessBot"
ENGINE_AUTHOR = "ChessBot-Battleground"


def board_to_obs(board: chess.Board) -> np.ndarray:
    """
    Encode a chess.Board to a canonical (8, 8) int8 observation array.
    Mirrors ChessEnv.get_piece_configuration() + canonical perspective flip.

    Positive values = current player's pieces, negative = opponent's.
    Piece type magnitudes: 1=pawn, 2=knight, 3=bishop, 4=rook, 5=queen, 6=king.
    """
    piece_map = np.zeros(64, dtype=np.int8)
    for sq, piece in board.piece_map().items():
        # +1 for white, -1 for black (before canonicalization)
        color_sign = 1 if piece.color == chess.WHITE else -1
        piece_map[sq] = piece.piece_type * color_sign

    state = piece_map.reshape(8, 8)
    # Canonical: negate if it's black's turn (so current player is always positive)
    return -state if board.turn == chess.BLACK else state


def load_model(model_name: str, weights_path: str, model_path: str | None, device: str):
    """Load a registered model with weights onto the given device.

    Temporarily redirects stdout to stderr during loading so that registry
    warnings don't pollute the UCI stdout channel.
    """
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        auto_register_models()
        if model_path:
            ModelRegistry._load_models_from_path(model_path)
        model = ModelRegistry.load_with_weights(model_name, weights_path)
    finally:
        sys.stdout = real_stdout

    model.to(device)
    model.eval()
    return model


def run_uci_loop(model, device: str, use_search: bool = False, num_sims: int = 100,
                 sample: bool = False, move_time: float = 0.1):
    """
    Main UCI protocol loop. Reads commands from stdin, writes to stdout.

    Supported commands: uci, isready, ucinewgame, position, go, quit, stop
    """
    board = chess.Board()

    # Lazy MCTS import only if needed
    mcts = None
    if use_search:
        import gym
        import adversarial_gym
        from chessbot.mcts import MonteCarloTreeSearch
        env = gym.make("Chess-v0")
        env.reset()
        mcts = MonteCarloTreeSearch(env, model)

    def log(msg: str):
        """Write a UCI response line."""
        print(msg, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        if line == "uci":
            log(f"id name {ENGINE_NAME}")
            log(f"id author {ENGINE_AUTHOR}")
            log("uciok")

        elif line == "isready":
            log("readyok")  # Subsequent isready calls (e.g. between games) are instant

        elif line == "ucinewgame":
            board.reset()
            if mcts is not None:
                env.reset()

        elif line.startswith("position"):
            # position startpos [moves m1 m2 ...]
            # position fen <fen> [moves m1 m2 ...]
            parts = line.split()
            idx = 1
            if parts[idx] == "startpos":
                board.reset()
                idx = 2
            elif parts[idx] == "fen":
                fen_parts = []
                idx = 2
                while idx < len(parts) and parts[idx] != "moves":
                    fen_parts.append(parts[idx])
                    idx += 1
                board.set_fen(" ".join(fen_parts))

            if idx < len(parts) and parts[idx] == "moves":
                for uci_move in parts[idx + 1:]:
                    board.push(chess.Move.from_uci(uci_move))

        elif line.startswith("go"):
            if use_search and mcts is not None:
                # Set board state in env, then run MCTS
                env.board = board.copy()
                obs = board_to_obs(board)
                fen = board.fen()
                action, _ = mcts.search(fen, (obs, None), num_simulations=num_sims)
            else:
                obs = board_to_obs(board)
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    log("bestmove 0000")
                    continue
                action, _ = model.get_action(obs, legal_moves, sample=sample)

            try:
                move = _action_to_move(board, action)
                log(f"bestmove {move.uci()}")
            except Exception:
                # Fallback: first legal move
                fallback = next(iter(board.legal_moves), None)
                if fallback:
                    log(f"bestmove {fallback.uci()}")
                else:
                    log("bestmove 0000")

        elif line in ("stop", "ponderhit"):
            pass  # No pondering support; ignore

        elif line == "quit":
            break


def main():
    parser = argparse.ArgumentParser(description="UCI wrapper for ChessBot neural network models")
    parser.add_argument("--model", required=True, help="Registered model name (e.g. sgu_chessbot)")
    parser.add_argument("--weights", required=True, help="Path to model weights (.safetensors or .bin)")
    parser.add_argument("--model-path", default=None, help="Extra directory to scan for model registrations")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--search", action="store_true", help="Use MCTS for move selection")
    parser.add_argument("--num-sims", type=int, default=100, help="MCTS simulations per move (default: 100)")
    parser.add_argument("--sample", action="store_true", help="Sample moves instead of greedy argmax")
    args = parser.parse_args()

    # Respond to uci immediately (before loading the model) so cutechess-cli
    # doesn't time out waiting for uciok. Model is loaded lazily on isready.
    for line in sys.stdin:
        line = line.strip()
        if line == "uci":
            print(f"id name {ENGINE_NAME}", flush=True)
            print(f"id author {ENGINE_AUTHOR}", flush=True)
            print("uciok", flush=True)
        elif line == "isready":
            break  # Load model now, then send readyok
        elif line == "quit":
            sys.exit(0)

    model = load_model(args.model, args.weights, args.model_path, args.device)
    print("readyok", flush=True)
    run_uci_loop(model, args.device, use_search=args.search, num_sims=args.num_sims, sample=args.sample)


if __name__ == "__main__":
    main()
