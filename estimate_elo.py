#!/usr/bin/env python3
"""
ELO estimation for ChessBot-Battleground neural network models.

Uses a Stockfish gauntlet: plays the model against Stockfish at various
calibrated ELO targets (via UCI_LimitStrength + UCI_Elo), finds the win-rate
crossover, and interpolates an estimated ELO.

Stockfish's UCI_Elo is calibrated at 120s+1s time control, anchored to CCRL 40/4.
Valid range: 1320–3190.

Usage:
    # In-process gauntlet (fast, default)
    python estimate_elo.py --model sgu_chessbot --weights path/to/model.safetensors

    # With custom ELO search range
    python estimate_elo.py --model sgu_chessbot --weights path/to/model.safetensors --elo-range 1500 2500

    # More games per bracket for accuracy
    python estimate_elo.py --model sgu_chessbot --weights path/to/model.safetensors --num-games 50

    # Enable MCTS
    python estimate_elo.py --model sgu_chessbot --weights path/to/model.safetensors --search --num-sims 100

    # Use cutechess-cli instead (generates PGN, industry-standard)
    python estimate_elo.py --model sgu_chessbot --weights path/to/model.safetensors --cutechess
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import chess
import chess.engine
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from chessbot.models.registry import ModelRegistry, auto_register_models
from chessbot.mcts.search import _move_to_action, _action_to_move

# ---------------------------------------------------------------------------
# Default ELO ladder to test against (subset of Stockfish's 1320–3190 range)
# ---------------------------------------------------------------------------
DEFAULT_ELO_LADDER = [1320, 1500, 1700, 1900, 2100, 2300, 2500, 2700, 2900, 3190]

STOCKFISH_PATH = os.path.join(os.path.dirname(__file__), "stockfish-x86_64-bmi2")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BracketResult:
    elo: int
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        """Score as a fraction [0, 1]: win=1, draw=0.5, loss=0."""
        if self.games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games

    @property
    def win_rate(self) -> float:
        """Win-only fraction (no draws counted as wins)."""
        if self.games == 0:
            return 0.0
        return self.wins / self.games

    def estimated_elo(self) -> Optional[float]:
        """ELO estimate from score using the standard formula.

        expected_score = 1 / (1 + 10^((opp_elo - player_elo) / 400))
        => player_elo = opp_elo - 400 * log10(1/score - 1)
        """
        s = self.score
        if s <= 0 or s >= 1:
            return None
        return self.elo - 400 * math.log10(1 / s - 1)

    def __str__(self) -> str:
        est = self.estimated_elo()
        est_str = f"~{est:.0f}" if est is not None else "N/A"
        return (f"vs ELO {self.elo:4d} | W:{self.wins:3d} D:{self.draws:3d} L:{self.losses:3d} "
                f"| score={self.score:.2f} | est_elo={est_str}")


# ---------------------------------------------------------------------------
# Board → observation (mirrors ChessEnv, no gym dependency)
# ---------------------------------------------------------------------------

def board_to_obs(board: chess.Board) -> np.ndarray:
    piece_map = np.zeros(64, dtype=np.int8)
    for sq, piece in board.piece_map().items():
        color_sign = 1 if piece.color == chess.WHITE else -1
        piece_map[sq] = piece.piece_type * color_sign
    state = piece_map.reshape(8, 8)
    return -state if board.turn == chess.BLACK else state


# ---------------------------------------------------------------------------
# Game loop (in-process, does NOT use the gym env)
# ---------------------------------------------------------------------------

def play_game_inprocess(
    model,
    sf_engine: chess.engine.SimpleEngine,
    model_plays_white: bool,
    move_time: float = 0.1,
    sample: bool = False,
    max_moves: int = 300,
    mcts=None,
    num_sims: int = 100,
) -> float:
    """
    Play a single game between the neural network model and Stockfish.
    Returns score from the MODEL's perspective: 1=win, 0.5=draw, 0=loss.
    """
    board = chess.Board()
    sf_limit = chess.engine.Limit(time=move_time)
    if mcts is not None:
        mcts.reset()

    for _ in range(max_moves):
        if board.is_game_over(claim_draw=True):
            break

        model_to_move = (board.turn == chess.WHITE) == model_plays_white

        if model_to_move:
            obs = board_to_obs(board)
            legal_moves = list(board.legal_moves)
            if mcts is not None:
                action, _ = mcts.search(board.fen(), (obs, None), num_simulations=num_sims)
            else:
                action, _ = model.get_action(obs, legal_moves, sample=sample)
            try:
                move = _action_to_move(board, action)
            except Exception:
                move = legal_moves[0]
        else:
            result = sf_engine.play(board, sf_limit)
            move = result.move

        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        # Max moves reached — count as draw
        return 0.5

    winner = outcome.winner
    if winner is None:
        return 0.5
    elif (winner == chess.WHITE) == model_plays_white:
        return 1.0
    else:
        return 0.0


def _progress_line(target_elo: int, game_num: int, num_games: int, result: BracketResult,
                   color: str, last_result: str) -> str:
    """Build a compact progress line for the current bracket."""
    score_str = f"{result.score:.2f}" if result.games > 0 else " --- "
    est = result.estimated_elo()
    est_str = f"~{est:.0f}" if est is not None else "  ?"
    return (f"  vs {target_elo} | Game {game_num:>{len(str(num_games))}}/{num_games} "
            f"[{color}] | W:{result.wins} D:{result.draws} L:{result.losses} "
            f"| score={score_str} | est={est_str}   ")


def run_bracket_inprocess(
    model,
    target_elo: int,
    num_games: int,
    move_time: float,
    sample: bool,
    mcts=None,
    num_sims: int = 100,
) -> BracketResult:
    """Play num_games against Stockfish at target_elo, alternating colors."""
    result = BracketResult(elo=target_elo)

    sf_path = STOCKFISH_PATH if os.path.exists(STOCKFISH_PATH) else "stockfish"
    sf_engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    sf_engine.configure({"UCI_LimitStrength": True, "UCI_Elo": target_elo})

    outcome_symbols = {1.0: "W", 0.5: "D", 0.0: "L"}

    try:
        for i in range(num_games):
            model_plays_white = (i % 2 == 0)
            color = "White" if model_plays_white else "Black"

            # Show progress before the game starts
            line = _progress_line(target_elo, i + 1, num_games, result, color, "")
            print(f"\r{line}", end="", flush=True)

            score = play_game_inprocess(model, sf_engine, model_plays_white, move_time, sample,
                                        mcts=mcts, num_sims=num_sims)

            if score == 1.0:
                result.wins += 1
            elif score == 0.5:
                result.draws += 1
            else:
                result.losses += 1

            # Update line with outcome of finished game
            last = outcome_symbols[score]
            line = _progress_line(target_elo, i + 1, num_games, result, color, last)
            print(f"\r{line}", end="", flush=True)
    finally:
        sf_engine.quit()

    # Clear progress line — summary row will be printed by the caller
    print("\r" + " " * 80 + "\r", end="", flush=True)
    return result


# ---------------------------------------------------------------------------
# Gauntlet: sweep ELO ladder, find crossover, interpolate
# ---------------------------------------------------------------------------

def elo_from_score(opponent_elo: float, score: float) -> Optional[float]:
    if score <= 0 or score >= 1:
        return None
    return opponent_elo - 400 * math.log10(1 / score - 1)


def interpolate_elo(low: BracketResult, high: BracketResult) -> tuple[float, float]:
    """
    Linear interpolation between two brackets where win-rate crosses 0.5.
    Returns (estimated_elo, uncertainty).
    """
    s_low = low.score
    s_high = high.score

    # Interpolate the ELO where score == 0.5
    if s_low == s_high:
        return (low.elo + high.elo) / 2, (high.elo - low.elo) / 2

    t = (0.5 - s_low) / (s_high - s_low)
    est = low.elo + t * (high.elo - low.elo)
    uncertainty = (high.elo - low.elo) / 2

    return est, uncertainty


def run_gauntlet(
    model,
    elo_ladder: list[int],
    num_games: int,
    move_time: float,
    sample: bool,
    verbose: bool = True,
    mcts=None,
    num_sims: int = 100,
) -> tuple[list[BracketResult], Optional[float], Optional[float]]:
    """
    Sweep through the ELO ladder and find the model's estimated ELO.

    Strategy:
    - Play games at each level in ascending order
    - Stop sweeping once we find two consecutive levels that bracket 50% score
    - Return all bracket results + interpolated estimate + uncertainty
    """
    results = []

    if verbose:
        header = f"{'ELO':>6}  {'W':>4} {'D':>4} {'L':>4}  {'Score':>6}  {'Est ELO':>9}"
        print(header)
        print("-" * len(header))

    prev = None
    final_estimate = None
    uncertainty = None

    for target_elo in elo_ladder:
        br = run_bracket_inprocess(model, target_elo, num_games, move_time, sample,
                                   mcts=mcts, num_sims=num_sims)
        results.append(br)

        est = br.estimated_elo()
        est_str = f"{est:9.0f}" if est is not None else "     N/A "

        if verbose:
            print(f"{target_elo:>6}  {br.wins:>4} {br.draws:>4} {br.losses:>4}  {br.score:>6.2f}  {est_str}")

        # Found crossover: prev had score > 0.5, current has score < 0.5
        if prev is not None and prev.score >= 0.5 > br.score:
            final_estimate, uncertainty = interpolate_elo(prev, br)
            # Continue to get one more data point if there's room, then stop
            break

        # If already below 0.5 on first bracket, model is below that ELO
        if prev is None and br.score < 0.5:
            # Estimate from this single bracket
            e = br.estimated_elo()
            if e is not None:
                final_estimate = e
                uncertainty = (target_elo - elo_ladder[0]) / 2 if len(elo_ladder) > 1 else 100
            break

        prev = br

    # If we swept the whole ladder without crossing 0.5, model is stronger than max
    if final_estimate is None and results:
        last = results[-1]
        if last.score > 0.5:
            e = last.estimated_elo()
            final_estimate = e
            uncertainty = 200  # High uncertainty: model may be stronger than tested

    return results, final_estimate, uncertainty


# ---------------------------------------------------------------------------
# cutechess-cli mode
# ---------------------------------------------------------------------------

def find_cutechess() -> Optional[str]:
    for name in ("cutechess-cli", "cutechess"):
        path = shutil.which(name)
        if path:
            return path
    return None


def parse_cutechess_output(output: str) -> tuple[int, int, int]:
    """Parse W/D/L from cutechess-cli stdout (final score line only)."""
    wins = draws = losses = 0
    # cutechess outputs lines like: "Score of MyBot vs Stockfish: 12 - 5 - 3  [0.613] 20"
    # Use findall and take the LAST match (final cumulative score)
    matches = re.findall(r"Score of MyBot vs \S+:\s+(\d+)\s+-\s+(\d+)\s+-\s+(\d+)", output)
    if matches:
        w, l, d = matches[-1]
        wins, losses, draws = int(w), int(l), int(d)
    return wins, draws, losses


def parse_pgn_results(pgn_path: str, game_count: int) -> tuple[int, int, int]:
    """Parse W/D/L for the last game_count games from a PGN file."""
    wins = draws = losses = 0
    results = []
    white_players = []
    with open(pgn_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("[White "):
                white_players.append("MyBot" if '"MyBot"' in line else "Stockfish")
            elif line.startswith("[Result "):
                m = re.search(r'"([^"]+)"', line)
                if m:
                    results.append(m.group(1))

    # Take only the last game_count games
    results = results[-game_count:]
    white_players = white_players[-game_count:]

    for result, white in zip(results, white_players):
        mybot_is_white = (white == "MyBot")
        if result == "1-0":
            if mybot_is_white:
                wins += 1
            else:
                losses += 1
        elif result == "0-1":
            if mybot_is_white:
                losses += 1
            else:
                wins += 1
        elif result == "1/2-1/2":
            draws += 1
        # "*" = unfinished, skip
    return wins, draws, losses


def run_bracket_cutechess(
    model_name: str,
    weights_path: str,
    model_extra_path: Optional[str],
    target_elo: int,
    num_games: int,
    move_time: float,
    cutechess_path: str,
    pgn_out: Optional[str] = None,
    debug: bool = False,
) -> BracketResult:
    """Run a bracket using cutechess-cli."""
    sf_path = STOCKFISH_PATH if os.path.exists(STOCKFISH_PATH) else "stockfish"

    # Build the uci_wrapper command
    wrapper_script = os.path.join(os.path.dirname(__file__), "uci_wrapper.py")
    wrapper_cmd = f"python {wrapper_script} --model {model_name} --weights {weights_path}"
    if model_extra_path:
        wrapper_cmd += f" --model-path {model_extra_path}"

    python_bin = sys.executable
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        cutechess_path,
        "-engine",
            f"cmd={python_bin}",
            f"arg={wrapper_script}",
            "arg=--model", f"arg={model_name}",
            "arg=--weights", f"arg={weights_path}",
            *(["arg=--model-path", f"arg={model_extra_path}"] if model_extra_path else []),
            f"dir={repo_dir}",
            "name=MyBot", "proto=uci", "timemargin=30000", "restart=off",
        "-engine",
            f"cmd={sf_path}",
            "name=Stockfish", "proto=uci",
            f"option.UCI_LimitStrength=true",
            f"option.UCI_Elo={target_elo}",
        "-each", "tc=40/30+1",
        *(["-debug"] if debug else []),
        "-games", str(num_games),
        "-repeat",
        "-maxmoves", "200",
    ]
    cmd += ["-pgnout", pgn_out or "games.pgn"]

    # Pass as shell string (cutechess-cli parses key=value tokens itself)
    # Pass current env explicitly so the venv python is found correctly
    shell_cmd = " ".join(f'"{t}"' if " " in t else t for t in cmd)
    print(f"\n[CMD] {shell_cmd}\n", flush=True)
    # Stream output live to terminal while also capturing for parsing
    output_lines = []
    with subprocess.Popen(shell_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, shell=True, env=os.environ.copy()) as proc:
        for line in proc.stdout:
            if debug:
                print(f"  [cutechess] {line}", end="", flush=True)
            output_lines.append(line)
    combined = "".join(output_lines)
    pgn_file = pgn_out or "games.pgn"
    if os.path.exists(pgn_file):
        wins, draws, losses = parse_pgn_results(pgn_file, num_games)
    else:
        wins, draws, losses = parse_cutechess_output(combined)
    return BracketResult(elo=target_elo, wins=wins, draws=draws, losses=losses)


# ---------------------------------------------------------------------------
# Output & reporting
# ---------------------------------------------------------------------------

def print_summary(
    model_name: str,
    results: list[BracketResult],
    estimate: Optional[float],
    uncertainty: Optional[float],
    mode: str,
):
    print()
    print("━" * 52)
    print(f"  Model:  {model_name}")
    print(f"  Mode:   {mode}")
    if estimate is not None:
        unc_str = f" ±{uncertainty:.0f}" if uncertainty is not None else ""
        print(f"  Estimated ELO: ~{estimate:.0f}{unc_str}")
        print(f"  (Calibrated to CCRL 40/4 via Stockfish UCI_Elo)")
    else:
        print("  Could not estimate ELO (not enough data)")
    print("━" * 52)


def save_results(
    model_name: str,
    results: list[BracketResult],
    estimate: Optional[float],
    uncertainty: Optional[float],
    args,
):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"elo_results_{model_name}_{ts}.json"
    data = {
        "model": model_name,
        "weights": args.weights,
        "timestamp": ts,
        "num_games_per_bracket": args.num_games,
        "move_time": args.move_time,
        "search": args.search,
        "estimated_elo": estimate,
        "uncertainty": uncertainty,
        "brackets": [
            {
                "elo": r.elo,
                "wins": r.wins,
                "draws": r.draws,
                "losses": r.losses,
                "score": r.score,
                "estimated_elo_from_bracket": r.estimated_elo(),
            }
            for r in results
        ],
    }
    with open(fname, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Estimate ELO of a ChessBot neural network model via Stockfish gauntlet"
    )
    parser.add_argument("--model", required=True,
                        help="Registered model name (e.g. sgu_chessbot)")
    parser.add_argument("--weights", required=True,
                        help="Path to model weights (.safetensors or .bin)")
    parser.add_argument("--model-path", default=None,
                        help="Extra directory to scan for model registrations")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="PyTorch device (default: cuda if available, else cpu)")
    parser.add_argument("--num-games", type=int, default=20,
                        help="Games per ELO bracket (default: 20)")
    parser.add_argument("--elo-range", type=int, nargs=2, default=None, metavar=("LOW", "HIGH"),
                        help="Min and max ELO to test (e.g. --elo-range 1500 2500)")
    parser.add_argument("--elo-ladder", type=int, nargs="+", default=None,
                        help="Explicit ELO values to test (overrides --elo-range)")
    parser.add_argument("--move-time", type=float, default=0.1,
                        help="Stockfish move time in seconds (default: 0.1)")
    parser.add_argument("--search", action="store_true",
                        help="Use MCTS for model move selection (slower)")
    parser.add_argument("--num-sims", type=int, default=100,
                        help="MCTS simulations per move if --search is set (default: 100)")
    parser.add_argument("--sample", action="store_true",
                        help="Sample moves from policy distribution instead of argmax")
    parser.add_argument("--cutechess", action="store_true",
                        help="Use cutechess-cli instead of in-process game loop")
    parser.add_argument("--pgn-out", default=None,
                        help="Save games to this PGN file (cutechess mode only)")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save results JSON")
    parser.add_argument("--debug", action="store_true",
                        help="Show cutechess-cli debug output (default: off)")
    args = parser.parse_args()

    # --- Determine ELO ladder ---
    if args.elo_ladder:
        elo_ladder = sorted(args.elo_ladder)
    elif args.elo_range:
        lo, hi = args.elo_range
        elo_ladder = [e for e in DEFAULT_ELO_LADDER if lo <= e <= hi]
        if not elo_ladder:
            elo_ladder = [lo, hi]
    else:
        elo_ladder = DEFAULT_ELO_LADDER

    # --- cutechess-cli check ---
    if args.cutechess:
        cutechess_path = find_cutechess()
        if not cutechess_path:
            print("Error: cutechess-cli not found on PATH.")
            print("Install with:  sudo apt install cutechess")
            print("Or download from: https://github.com/cutechess/cutechess/releases")
            sys.exit(1)

    # --- Load model (not needed for cutechess mode — uci_wrapper loads it) ---
    model = None
    if not args.cutechess:
        print(f"Loading model '{args.model}' from {args.weights} ...", end=" ", flush=True)
        auto_register_models()
        if args.model_path:
            ModelRegistry._load_models_from_path(args.model_path)
        model = ModelRegistry.load_with_weights(args.model, args.weights)
        model.to(args.device)
        model.eval()
        print("done.")

    # --- MCTS setup (in-process mode only) ---
    mcts = None
    if args.search and not args.cutechess:
        from chessbot.mcts.search import MonteCarloTreeSearchFast

        # MonteCarloTreeSearchFast uses a live chess.Board internally — no gym env needed.
        # It takes (fen, (obs, None)) directly, so the correct position is always used.
        mcts = MonteCarloTreeSearchFast(game_state=None, nnet=model)

    # --- Print run info ---
    mode = "cutechess-cli" if args.cutechess else "in-process"
    print(f"\nELO Gauntlet: {args.model} vs Stockfish (UCI_LimitStrength)")
    print(f"Games/bracket: {args.num_games} | Move time: {args.move_time}s | "
          f"MCTS: {'on' if args.search else 'off'} | Mode: {mode}")
    print(f"ELO ladder: {elo_ladder}")
    print()

    # --- Auto-name PGN file if not specified ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pgn_out = args.pgn_out or f"games_{args.model}_{ts}.pgn"

    # --- Run gauntlet ---
    if args.cutechess:
        results = []
        header = f"{'ELO':>6}  {'W':>4} {'D':>4} {'L':>4}  {'Score':>6}  {'Est ELO':>9}"
        print(header)
        print("-" * len(header))

        prev = None
        final_estimate = None
        uncertainty = None

        for target_elo in elo_ladder:
            print(f"Testing vs ELO {target_elo}...", end=" ", flush=True)
            br = run_bracket_cutechess(
                args.model, args.weights, args.model_path,
                target_elo, args.num_games, args.move_time,
                cutechess_path, pgn_out, debug=args.debug
            )
            results.append(br)

            est = br.estimated_elo()
            est_str = f"{est:9.0f}" if est is not None else "     N/A "
            print(f"\r{target_elo:>6}  {br.wins:>4} {br.draws:>4} {br.losses:>4}  {br.score:>6.2f}  {est_str}")

            if prev is not None and prev.score >= 0.5 > br.score:
                final_estimate, uncertainty = interpolate_elo(prev, br)
                break
            if prev is None and br.score < 0.5:
                e = br.estimated_elo()
                if e is not None:
                    final_estimate = e
                    uncertainty = 150
                break
            prev = br

        if final_estimate is None and results:
            last = results[-1]
            if last.score > 0.5:
                final_estimate = last.estimated_elo()
                uncertainty = 200
    else:
        results, final_estimate, uncertainty = run_gauntlet(
            model, elo_ladder, args.num_games, args.move_time, args.sample,
            mcts=mcts, num_sims=args.num_sims,
        )

    # --- Summary ---
    print_summary(args.model, results, final_estimate, uncertainty, mode)

    if not args.no_save:
        save_results(args.model, results, final_estimate, uncertainty, args)


if __name__ == "__main__":
    main()
