"""
Used to generate PGN files from the downloadable Lichess puzzle database in CSV format.
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

# Example usage
input_csv_path = '/home/kage/chess_workspace/PGN_dataset/pgn-data/Puzzles/LichessPuzzles-csv/lichess_db_puzzle-jan2-2025.csv'
output_directory = '/home/kage/chess_workspace/PGN_dataset/pgn-data/Puzzles/LichessPuzzles-csv/jan-2025-pgns/'
lichess_csv_to_pgn(input_csv_path, output_directory, chunk_size=50_000)