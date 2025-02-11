"""
This section loads the collected PGN files and distributes them across num_output_files PGN files. 

Distributes by "dealing" each game to a different output file in a circular fashion.
"""
from itertools import cycle
import random, os
import glob
import chess.pgn

def distribute_games_across_files(filepaths, output_dir, num_output_files):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_files = [open(os.path.join(output_dir, f'chesspgn_{i}.pgn'), 'w', encoding='utf-8') for i in range(num_output_files)]
    output_file_cycle = cycle(output_files)  # Create a cycle of output files for round-robin distribution

    for i, filepath in enumerate(filepaths):
        print(f"{i}/{len(filepaths)}   - filepath: {filepath} ")
        with open(filepath, 'r', encoding='utf-8') as pgn_file:
            while game := chess.pgn.read_game(pgn_file):
                output_file = next(output_file_cycle) 
                exporter = chess.pgn.FileExporter(output_file)
                game.accept(exporter)

    for f in output_files:
        f.close()


def process_pgn_files(input_dir, output_dir, num_output_files):
    filepaths = list(glob.iglob(f'{input_dir}/**/*.pgn', recursive=True))
    random.shuffle(filepaths)  # Shuffle the list of file paths
    distribute_games_across_files(filepaths, output_dir, num_output_files)


"""
Split into test and train
"""
import shutil

def split_pgn_files(input_dir, train_dir, test_dir, train_ratio=0.9):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    filepaths = list(glob.iglob(f'{input_dir}/*.pgn'))
    random.shuffle(filepaths)
    
    split_index = int(len(filepaths) * train_ratio)
    
    train_files = filepaths[:split_index]
    test_files = filepaths[split_index:]
    
    for filepath in train_files:
        shutil.move(filepath, os.path.join(train_dir, os.path.basename(filepath)))
    
    for filepath in test_files:
        shutil.move(filepath, os.path.join(test_dir, os.path.basename(filepath)))

"""
Generate PGN files from the downloadable Lichess puzzle database, which comes in CSV format.
"""
import pandas as pd
import chess
import chess.pgn
import os


def lichess_csv_to_pgn(input_csv_path, output_dir, chunk_size=1000):
    """ Convert the puzzle csv file from lichess open database into a pgn file"""

    os.makedirs(output_dir, exist_ok=True)

    def generate_pgn(row):
        board = chess.Board(row['FEN'])
        
        moves = row['Moves'].split(' ')

        first_move = chess.Move.from_uci(moves[0])
        board.push(first_move)

        game = chess.pgn.Game()
        game.setup(board)
        starting_fen = board.fen()

        winner = 'White to move.' if board.turn == chess.WHITE else 'Black to move.'
        result = "1-0" if board.turn == chess.WHITE else "0-1"

        game.headers['Site'] = row['GameUrl']
        game.headers['ToMove'] = winner
        game.headers['Opening'] = str(row['OpeningTags'])
        game.headers['Rating'] = str(row['Rating'])
        game.headers['Themes'] = str(row['Themes'])
        game.headers['Result'] = result  
        game.headers['FEN'] = starting_fen

        node = game
        for move in moves[1:]:
            uci_move = chess.Move.from_uci(move)
            if uci_move in board.legal_moves:
                board.push(uci_move)
                node = node.add_variation(uci_move)
            else:
                return "Invalid move sequence"
        
        return str(game) + '\n\n'
        

    # Read the CSV file in chunks
    chunk_iterator = pd.read_csv(input_csv_path, chunksize=chunk_size)
    
    for i, chunk in enumerate(chunk_iterator):
        print(f"Processing chunk {i + 1}")
        pgn_filename = os.path.join(output_dir, f"puzzles_{i + 1}.pgn")
        
        chunk_pgn_content = chunk.apply(generate_pgn, axis=1).str.cat()
        
        with open(pgn_filename, "w") as file:
            print(chunk_pgn_content, file=file)
