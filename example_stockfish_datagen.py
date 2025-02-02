# from chessbot.data import solve_and_export

# # Example 1: Solve a PGN file 
# pgn_file = '/home/kage/chess_workspace/PGN_dataset/pgn-data/Puzzles/bill_wall_puzzles/analyze/useFish-Shumilin - Chess Tactics Training.pgn'
# output_file = '/home/kage/chess_workspace/PGN_dataset/pgn-data/Puzzles/bill_wall_puzzles/analyze_out/useFish-Shumilin - Chess Tactics Training-fished.pgn'

# solve_and_export(
#     pgn_file=pgn_file, 
#     output_file=output_file, 
#     num_proc=5,                 # Number of processes to run
#     threads=3,                  # Number of stockfish threads to search with
#     hash_size=1024*4,
#     max_moves=15                # Maximum number of moves to make from position
# )


# # # Example 2: Solve directory of PGN files
# #
# pgn_dir = '/home/kage/chess_workspace/PGN_dataset/pgn-data/Puzzles/bill_wall_puzzles/analyze'
# output_dir = '/home/kage/chess_workspace/PGN_dataset/pgn-data/Puzzles/bill_wall_puzzles/analyze_out'

# solve_and_export(
#     pgn_dir=pgn_dir, 
#     output_dir=output_dir, 
#     num_proc=5,                 # Number of processes to run
#     threads=3,                  # Number of stockfish threads to search with
#     hash_size=1024*4,
#     max_moves=15                # Maximum number of moves to make from position
# )


# # # Example 3: Randomly solve positions from games in a PGN file
# #
# from chessbot.data import solve_and_export_random_positions 

# pgn_file = '/home/kage/chess_workspace/PGN_dataset/pgn-data/GM-games/TWIC-FR/wchfischerko22.pgn'
# output_dir = '/home/kage/chess_workspace/PGN_dataset/pgn-data/GM-games/TWIC-FR'

# solve_and_export_random_positions(
#     pgn_path=pgn_file, 
#     output_dir=output_dir,
#     max_moves=15,                # Maximum number of moves to make from position
#     p=0.25,                      # Probability of solving a position
#     fr=True,                    # Fischer Random 
#     num_proc=5,
#     threads=6,                   # Number of stockfish threads to search with
#     hash_size=1024*3,
# )


# # Example 4: Annotate a PGN file with stockfish evaluations
#
from chessbot.data import annotate_and_export

pgn_dir = '/home/kage/chess_workspace/PGN_dataset/pgn-data/GM-games/TWIC-FR/'
output_dir = '/home/kage/chess_workspace/PGN_dataset/pgn-data/GM-games/TWIC-FR/'
fish_path = '/home/kage/chess_workspace/chess-puzzle-maker/stockfish-x86_64-bmi2'

annotate_and_export(
    pgn_path=pgn_dir,
    output_dir=output_dir,
    fish_path=fish_path,
    fr=True,
    num_proc=5,
    threads=6,
    hash_size=1024*3,
)