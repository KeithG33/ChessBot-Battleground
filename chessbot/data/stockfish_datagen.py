""" 
Randomly select positions from PGN games and evaluate max_moves moves from each position. 
"""

import os
import sys
import time
import io
import random
from typing import Optional
import chess
import chess.pgn
from stockfish import Stockfish
from multiprocessing import Pool
from functools import partial


# Global variable for Stockfish in each worker
worker_stockfish = None


def is_fen(fen: str) -> bool:
    """
    Check if the given string is a valid FEN.
    """
    try:
        board = chess.Board(fen)
        if not board.is_valid():
            return False
        return True
    except ValueError:
        return False


def read_games(pgn_file: str) -> list[str]:
    """
    Read all games from a PGN file.
    """
    with open(pgn_file, 'r', encoding='utf-8') as f:
        games = []
        while game := chess.pgn.read_game(f):
            exporter = chess.pgn.StringExporter(
                headers=True, variations=True, comments=True
            )
            games.append(game.accept(exporter))
    return games


def wdl_to_value(wdl: tuple[int, int, int]) -> float:
    """Converts win-draw-loss statistics to a value between -1 and 1."""
    win, draw, loss = wdl
    total = win + draw + loss
    score = win + 0.5 * draw
    score = score / total
    score = score * 2 - 1
    return score


class StockfishSolver:
    """
    Class to solve chess positions using Stockfish.
    """

    max_moves: int = 20
    move_time: int = 0  # milliseconds

    def __init__(
        self,
        stockfish: Optional[Stockfish] = None,
        max_moves: int = 20,
        move_time: int = 1000,
    ):
        self.stockfish = stockfish
        self.max_moves = max_moves
        self.move_time = move_time

    @staticmethod
    def evaluate_position(
        stockfish, move_time=None
    ) -> tuple[None, None, None] | tuple[chess.Move, float]:
        """
        Evaluate a position with Stockfish and return the best move and score.
        """

        # TODO: investigate stockfish.get_best_move_time(move_time) - doesn't do anything for me
        move_uci = stockfish.get_best_move()

        if move_uci is None:
            return None, None

        move = chess.Move.from_uci(move_uci)

        wdl = stockfish.get_wdl_stats()
        score = wdl_to_value(wdl)

        return move, score

    def solve_fen(self, fen: str, headers: dict = {}) -> str:
        """
        Solve a position given its FEN and return a PGN string with the solution.

        Note the result is changed to "*", and a stockfish evaluation is added as a comment. The
        stockfish eval replaces the result as the value of the position
        """

        board = chess.Board(fen)
        self.stockfish.set_fen_position(fen)

        # Create a new game with headers
        game = chess.pgn.Game().from_board(board)
        game.headers.update(headers)
        game.headers["SetUp"] = "1"
        game.headers["Result"] = "*"
        game.headers["FEN"] = fen

        event = game.headers.get("Event", "")
        game.headers["Event"] = f"{event} - SF Selfplay"

        # Make first move to get first child node
        move, score = self.evaluate_position(self.stockfish)

        if move is None:
            return ""

        node = game.add_variation(move)
        node.comment = f"sf:,{score:.2f}"

        self.stockfish.make_moves_from_current_position([move.uci()])
        board.push(move)

        # Stockfish self-play
        for _ in range(self.max_moves - 1):
            if board.is_game_over():
                break

            move, score = self.evaluate_position(self.stockfish)

            if move is None:
                break

            node = node.add_variation(move)
            node.comment = f"sf:,{score:.2f}"

            self.stockfish.make_moves_from_current_position([move.uci()])
            board.push(move)

        exporter = chess.pgn.StringExporter(
            headers=True, variations=True, comments=True
        )
        return game.accept(exporter)

    def solve_position(self, game: chess.pgn.Game):
        """
        Solve the given game position and return the solution as a PGN string.

        See self.solve_fen for details.
        """
        fen = game.board().fen()
        headers = game.headers

        solution = self.solve_fen(fen, headers)
        return solution


class StockfishAnnotator:
    def __init__(self, stockfish: Stockfish, move=True, score=True):
        self.stockfish = stockfish
        self.move = move
        self.score = score


    def evaluate_position(self, board):
        """
        Evaluate a position with stockfish. 
        """
        score = ""
        move_san = ""
        if self.move:
            move_uci = self.stockfish.get_best_move()
            move_san = board.san(chess.Move.from_uci(move_uci))

        if self.score:
            wdl = self.stockfish.get_wdl_stats()
            score = wdl_to_value(wdl)

        return move_san, str(score)
    
    def process_game(self, game):
        """
        Process game, add comments, return the game
        """
        board = game.board()
        self.stockfish.set_fen_position(board.fen())

        for i, node in enumerate(game.mainline()):
            board.push(node.move)
            t1 = time.perf_counter()
            move, score = self.evaluate_position(board)
            comment = f"sf:{move},{score}"
            node.comment = comment
        
        return game
    
    def annotate_pgn(self, pgn_file, output_file):
        """
        Annotate a pgn file with stockfish evaluations
        """
        with open(pgn_file) as pgn:
            with open(output_file, "w") as output:
                while (game := chess.pgn.read_game(pgn)):
                    game = self.process_game(game)
                    print(game, file=output, end="\n\n")
                    print(f"Annotated game {game.headers['Event']}")



def initialize_worker(
    fish_path: str = None, params: dict = None, depth: int = 27, fr: bool = False
):
    """
    Initializer for each worker process to create its own Stockfish instance. Used in multiprocessing.
    """
    global worker_stockfish
    if fish_path is None:
        fish_path = (
            "stockfish-x86_64-bmi2"
        )

    if params is None:
        params = {"Threads": 2, "Hash": 1024 * 2}

    if fr:
        params["UCI_Chess960"] = True

    worker_stockfish = Stockfish(path=fish_path, parameters=params, depth=depth)


def dispatch_fen_solver(task: tuple) -> tuple:
    """
    Worker function to solve a specific position. Used in multiprocessing.
    Args:
        task (tuple): A tuple containing (output_file, FEN, max_moves)
    Returns:
        tuple: A tuple containing (output_file, solution_pgn)
    """
    global worker_stockfish
    if worker_stockfish is None:
        raise ValueError("Stockfish instance not initialized in worker.")

    fen, max_moves = task
    solver = StockfishSolver(worker_stockfish, max_moves=max_moves)
    solution_pgn = solver.solve_fen(fen)
    return solution_pgn


def dispatch_game_solver(task: tuple[str, int]) -> str:
    """
    Worker function to solve a specific position from a game/puzzle. Used in multiprocessing.
    """
    global worker_stockfish
    if worker_stockfish is None:
        raise ValueError("Stockfish instance not initialized in worker.")

    game_pgn_str, max_moves = task
    solver = StockfishSolver(worker_stockfish, max_moves=max_moves)

    if is_fen(game_pgn_str):
        return solver.solve_fen(game_pgn_str)

    if not (game := chess.pgn.read_game(io.StringIO(game_pgn_str))):
        return ""

    solved_game = solver.solve_position(game)
    return solved_game


def dispatch_annotator(task: str) -> str:
    """
    Worker function to annotate a game with Stockfish evaluations. Used in multiprocessing.
    """
    global worker_stockfish
    if worker_stockfish is None:
        raise ValueError("Stockfish instance not initialized in worker.")
    
    annotator = StockfishAnnotator(worker_stockfish, move=False, score=True)
    game = annotator.process_game(chess.pgn.read_game(io.StringIO(task)))
    
    return str(game)

def solve_and_export(
    pgn_path: str,
    output_dir: str,
    max_moves: int = 20,
    fr: bool = False,
    num_proc: int = 5,
    threads: int = 7,
    hash_size: int = 1024 * 2,
):
    """Solves the pgn files within a directory using multiprocessing

    Creates a PGN file with Stockfish moves, and evaluations as comments
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.isfile(pgn_path) and pgn_path.endswith(".pgn"):
        pgn_files = [pgn_path]
    else:
        pgn_files = [f.path for f in os.scandir(pgn_path) if f.name.endswith(".pgn")]

    params = {"Threads": threads, "Hash": hash_size}
    params["UCI_Chess960"] = True if fr else False

    worker_init = partial(initialize_worker, params=params)
    with Pool(processes=num_proc, initializer=worker_init) as pool:
        for pgn_file in pgn_files:
            output_file = os.path.join(output_dir, os.path.basename(pgn_file))
            output_file = output_file.replace(".pgn", "-fished.pgn")

            games = read_games(pgn_file)

            tasks = [(game, max_moves) for game in games]

            with open(output_file, 'w', encoding='utf-8') as out_f:
                for idx, annotated_game in enumerate(
                    pool.imap_unordered(dispatch_game_solver, tasks), 1
                ):
                    if annotated_game:
                        print(annotated_game, file=out_f, end="\n\n")
                        print(f"Done processing game {idx}")

            print(f"Annotated file written to: {output_file}")


def annotate_and_export(
    pgn_path: str,
    output_dir: str,
    fish_path: str,
    fr: bool = False,
    num_proc: int = 1,
    threads: int = 2,
    hash_size: int = 1024
):
    """
    Annotate a pgn file with stockfish evaluations
    """
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(pgn_path) and pgn_path.endswith(".pgn"):
        pgn_files = [pgn_path]
    else:
        pgn_files = [f.path for f in os.scandir(pgn_path) if f.name.endswith(".pgn")]

    params = {"Threads": threads, "Hash": hash_size}
    params["UCI_Chess960"] = True if fr else False

    worker_init = partial(initialize_worker, fish_path=fish_path, params=params, fr=fr)
    with Pool(processes=num_proc, initializer=worker_init) as pool:
        for pgn_file in pgn_files:
            output_file = os.path.join(output_dir, os.path.basename(pgn_file))
            output_file = output_file.replace(".pgn", "-fished-annos.pgn")

            games = read_games(pgn_file)

            with open(output_file, 'w', encoding='utf-8') as out_f:
                for idx, annotated_game in enumerate(
                    pool.imap_unordered(dispatch_annotator, games), 1
                ):
                    if annotated_game:
                        print(annotated_game, file=out_f, end="\n\n")
                        print(f"Done processing game {idx}")


def solve_and_export_random_positions(
    pgn_path: str,
    output_dir: str,
    max_moves: int = 20,
    p: float = 0.1,  # Probability to solve a position
    fr: bool = False,
    num_proc: int = 5,
    threads: int = 1,
    hash_size: int = 1024 * 2,
):
    """
    Randomly solve positions within PGN games with probability p, using Stockfish, and export
    the sequences to pgn files

    """
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(pgn_path) and pgn_path.endswith(".pgn"):
        pgn_files = [pgn_path]
    else:
        pgn_files = [f.path for f in os.scandir(pgn_path) if f.name.endswith(".pgn")]
    
    params = {"Threads": threads, "Hash": hash_size}
    params["UCI_Chess960"] = True if fr else False

    # Use the top-level initialize_worker function with partial
    worker_init = partial(
        initialize_worker, fr=fr, params=params
    )

    with Pool(processes=num_proc, initializer=worker_init) as pool:
        for i, pgn_file in enumerate(pgn_files):
            print(f"Processing file: {pgn_file}")
            output_file = os.path.join(output_dir, os.path.basename(pgn_file))
            output_file = output_file.replace(".pgn", "-fished.pgn")

            if os.path.exists(output_file):
                mode = "a" # Append if file exists

            games = read_games(pgn_file)

            # Iterate through each game and select random positions
            solution_tasks = []
            for game_str in games:
                game = chess.pgn.read_game(io.StringIO(game_str))
                if game is None:
                    continue

                board = game.board()
                for move in game.mainline_moves():
                    board.push(move)

                    # Solve with some probability
                    if random.random() < p:
                        fen = board.fen()
                        solution_tasks.append((fen, max_moves))

            if not solution_tasks:
                print(f"No positions selected for solving")
                continue

            print(f"Selected {len(solution_tasks)} positions for solving")

            for idx, solution_pgn in enumerate(
                pool.imap_unordered(dispatch_fen_solver, solution_tasks), 1
            ):
                if solution_pgn:
                    with open(output_file, mode=mode, encoding='utf-8') as out_f:
                        out_f.write(solution_pgn + "\n\n")

                    print(f"Done processing solution {idx}")

            print(f"Annotated solutions written to: {output_file}")


# if __name__ == "__main__":

#     pgn_directory = "/home/kage/chess_workspace/stockfish_datagen/datagen"
#     output_directory = "/home/kage/chess_workspace/stockfish_datagen/datagen"

#     threads = 7  # Number of threads to use
#     num_proc = 2  # Number of processes to use
#     probability = 0.1  # 10% chance to solve any given position
#     max_solve_moves = 11  # Maximum number of moves to solve

#     solve_and_export_random_positions(
#         pgn_dir=pgn_directory,
#         output_dir=output_directory,
#         threads=threads,
#         hash_size=1024*4,
#         num_proc=num_proc,
#         p=probability,
#         max_moves=max_solve_moves
#     )
