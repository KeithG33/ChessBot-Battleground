# # Description: This file is basically just to store a collection of scripts for pre-processing the data
# #              coming from a few different sources.

"""For parsing games in lumbras database (games not puzzles. See separate file for puzzles)"""
import os
import chess
import chess.pgn

SPECIAL_CHARS = ['?', '??', '', ' ']

def condition_met(headers, rating_threshold=2600):
    """ Return True if the game should be saved, False otherwise."""

    source = headers.get("Source", "")
    if source == "LichessEliteDatabase":
        # Rapid/Classical only - might help filter high lichess ELOs 
        time_control = headers.get("Event")
        if "Blitz" in time_control:
            return False

    header_w = headers.get("WhiteElo", 0)
    header_b = headers.get("BlackElo", 0)

    if header_w in SPECIAL_CHARS:
        header_w = 0
    if header_b in SPECIAL_CHARS:
        header_b = 0
    
    white_elo = int(header_w)
    black_elo = int(header_b)

    # If both ELO are in pgn headers then be strict
    if white_elo and black_elo:
        rating_cond = (white_elo > rating_threshold and black_elo > rating_threshold)

    # If only one ELO is in pgn headers then be lenient
    elif white_elo or black_elo:
        rating_cond = (white_elo > rating_threshold or black_elo > rating_threshold)
    
    else:
        rating_cond = False

    return rating_cond

def save_filtered_games(input_pgn_path, output_pgn_path):
    # Track games that meet the conditions
    game_offsets = []

    print(f"Processing {input_pgn_path}")
    with open(input_pgn_path, "r", encoding="ISO-8859-1") as pgn:
        while True:
            offset = pgn.tell()
            headers = chess.pgn.read_headers(pgn)

            if headers is None:
                break

            if condition_met(headers):
                game_offsets.append(offset)
            
    # Read and write games that meet conditions
    with open(input_pgn_path, "r", encoding="ISO-8859-1") as pgn:
        with open(output_pgn_path, "w", encoding="utf-8") as output_pgn:
            exporter = chess.pgn.FileExporter(output_pgn)
            for offset in game_offsets:
                pgn.seek(offset)  # Move to the game's offset
                game = chess.pgn.read_game(pgn)  # Read the full game
                game.accept(exporter)  # Save it
    
    print(f"DONE: Saved {len(game_offsets)} games to {output_pgn_path}")

def filter_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.scandir(input_dir):
        if not file.name.endswith(".pgn"):
            continue

        name = os.path.splitext(file.name)[0]
        input_pgn_path = os.path.join(input_dir, file.name)
        output_pgn_path = os.path.join(output_dir, f"{name}-2600+.pgn")
        save_filtered_games(input_pgn_path, output_pgn_path)


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
import chess
import chess.pgn
import os

def lichess_csv_to_pgn(input_csv_path, output_dir, chunk_size=1000):
    """ Convert the puzzle csv file from lichess open database into a pgn file"""

    import pandas as pd
    
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
