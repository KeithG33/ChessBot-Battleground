""" Use stockfish to analyze each position in a pgn file """

import os
import sys
import time

import chess
import chess.pgn

from stockfish import Stockfish


def wdl_to_value(wdl):
    """
    Converts win-draw-loss statistics to a value between -1 and 1
    """
    win, draw, loss = wdl
    
    total = win + draw + loss
    score = win + 0.5 * draw

    score = score / total
    score = score * 2 - 1
    
    return score


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
            print(f"Evaluated move {i + 1} in {time.perf_counter() - t1:.4f} seconds")
            # print(f"Evaluated move {i + 1}: {move} {score}")
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


def annotate_and_export(
    pgn_path: str,
    output_dir: str,
    fish_path: str,
    fr: bool = False,
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

    stockfish = Stockfish(
        fish_path,
        parameters=params,
        depth=27
    )

    annotator = StockfishAnnotator(stockfish, move=False, score=True)

    for pgn_file in pgn_files:
        output_file = os.path.join(output_dir, os.path.basename(pgn_file))
        output_file = output_file.replace(".pgn", "-fished-score.pgn")
        annotator.annotate_pgn(pgn_file, output_file)
        print(f"Annotated {pgn_file} - saved to {output_file}")

    stockfish.quit()


PGN_PATH = '/home/kage/chess_workspace/PGN_dataset/test.pgn'
FISH_PATH = '/home/kage/chess_workspace/chess-puzzle-maker/stockfish-x86_64-bmi2'

output_dir = '/home/kage/chess_workspace/PGN_dataset'
annotate_and_export(PGN_PATH, output_dir, FISH_PATH, threads=1, hash_size=2048)


