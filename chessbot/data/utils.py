"""
This section loads the collected PGN files and then distributes them
across the selected num_output_files PGN files. 

Ensures each PGN file has a wide spread of data.
"""
from itertools import cycle
import random, os
import glob
import chess.pgn

def get_pgn_file_paths(input_dir):
    return list(glob.iglob(f'{input_dir}/**/*.pgn', recursive=True))

def distribute_games_across_files(filepaths, output_dir, num_output_files):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare output files for writing
    output_files = [open(os.path.join(output_dir, f'chesspgn_{i}.pgn'), 'w', encoding='utf-8') for i in range(num_output_files)]
    output_file_cycle = cycle(output_files)  # Create a cycle of output files for round-robin distribution

    # game_count = 0
    for i, filepath in enumerate(filepaths):
        print(f"{i}/{len(filepaths)}   - filepath: {filepath} ")
        with open(filepath, 'r', encoding='utf-8') as pgn_file:
            while game := chess.pgn.read_game(pgn_file):
                # game_count += 1
                output_file = next(output_file_cycle)  # Get the next file in the cycle
                exporter = chess.pgn.FileExporter(output_file)
                game.accept(exporter)
    
    # Close all output files
    for f in output_files:
        f.close()

    # print(f"Total games: {game_count}")

def process_pgn_files(input_dir, output_dir, num_output_files):
    filepaths = get_pgn_file_paths(input_dir)
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

