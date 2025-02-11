# ChessBot Battleground

### 🚧 Under Construction 🚧

Project under development. Stay tuned!



##  Introduction

This repository has two main purposes: store a gigantic curated ChessBot-Dataset for ML training, and provide some simple Python/PyTorch infrastructure to quickly design, train, and compare model architectures. See how well your model architecture ideas can extract information from tiny 8x8 chess board images. 

This library provides support for:

- **Dataset/Training:** PyTorch dataset and training code
- **Gym Environment:** Gym environment for inference and self-play
- **Dueling:** Functionality for best-of-N matches between saved models
- **MCTS:** Simple implementation to give your supervised models search capability (*with training coming soon*).
- **Visualization:** Watch your models in action as they play and adapt on the board.

Write a model that subclasses `BaseChessModel` and take advantage of any of the available features. See one in action below:

<div align="center"  id="chess-battle-gif">
  <img src="assets/chessSGU-R8.1-selfplay.gif" style="width: 35%; height: auto;">
  <p><em>Self-play after a few days of training a 300M parameter network</em></p>
</div>



## 📂 Dataset
*[Releases available for download here](https://github.com/KeithG33/ChessBot-Battleground/releases)*  

Currently the dataset contains approximately **700 million positions** in **PGN format**, split across 1000 files, using an aggressive 90 train / 10 test split for learning ;) . Huge credits to the following main sources:

- Lumbra's Database (filtered 2600+)
- Lichess Puzzle Database
- Computer Chess: TCEC Database, CCRL
- Includes chess960/FR/freestyle
- (*coming soon*) Stockfish Data: position evaluation, puzzle solutions, best-move sequences  
  
      
The PyTorch `ChessDataset` is provided in `chessbot/data/dataset.py` to load the data for training.

```python
import chessbot.data.ChessDataset

pgn_files   = 'path/to/pgn_files' # Can be file, directory, or list of files
num_threads = 8

dataset    = ChessDataset(pgn_files, num_threads=num_threads)
dataloader = DataLoader(dataset, batch_size=bsz)

# Get example batch
batch = next(iter(dataloader))
states, actions, results = batch # (B, 8, 8),  (B, 4672),  (B,)
```



## 🧠 Training

A `ChessTrainer` class can be used to train different models with a variety of configurations. As a way to be flexible to a wide range of compute powers, the trainer splits the data loading and training into rounds and epochs. Each round will sample a new subset of `cfg.dataset.size_train` files, and then perform epochs on this subset.   

Each of the raw files in the data has been uniformly spread into the training dataset so sampling subsets still yields a highly varied dataset that can give good performance. Some more features of the trainer class

**ChessTrainer**
- HuggingFace's `accelerate` for easily adding mixed precision, compilation, gradient clipping, gradient accumulation
- Warmup LR scheduling and decay (linear or cosine)
- Configurable validation frequency in iterations
- Optional logging to wandb 


Here's a simplified example of using it:

```python
from chessbot.config import get_config()
from chessbot.train import ChessTrainer
from chessnet import YourChessNet

# Get default cfg and do some basic setup
cfg = get_config() # get default cfg

cfg.dataset.train_path = 'ChessBot-Battleground/dataset/'
cfg.dataset.size_train = 25
cfg.dataset.size_test = 5
cfg.train.batch_size = 128
cfg.train.lr = 0.001
cfg.train.output_dir = 'output/'

model = YourChessNet()

trainer = ChessTrainer(cfg, model)
trainer.train()
```

Please see the config in `chessbot/train/config.yaml` for a list and description of the available options. Also see the [Getting Started](#getting-started) for a quickstart on model format.

## 🤖 Inference & Battling

Take your models to the battleground!  

The library includes an **Adversarial Gym Environment** designed for two-player turn-based games, that can be used to visualize model inference. Watch your models in a few lines of code:

```python
import gym
import adversarial_gym
from example_model.simple_chessnet import SimpleChessNet

env = gym.make('Chess-v0', render_mode='human')
model = SimpleChessNet()
observation, info = env.reset()
done = False

while not done:
    legal_moves = env.board.legal_moves
    action = model.get_action(observation[0], legal_moves)
    observation, reward, done, truncated, info = env.step(action)
```

Code taken from `ChessBot-Battleground/examples/example_inference.ipynb`. Similar code also exists to harness **Monte Carlo Tree Search (MCTS)** for search during inference. *MCTS training code coming soon!*

The [Chess Battle GIF](#chess-battle-gif) at the beginning is an example of rendering the game with the Chess-env, and using MCTS for model inference. 

## 📈 Future Plans

- Clean and deduplicate the dataset
- Expand the dataset, add Stockfish generated data, get to an epic milestone of **1 billion positions**.  
- Release **MCTS training pipelines**.  
- Add enhanced tools for training and visualization.


## Getting Started
**1. Installation:**  

Before installing the library, first install the [Adversarial Gym](https://github.com/OperationBeatMeChess/adversarial-gym) chess environment:
  ```bash
  pip install adversarial-gym
  ```
Now install ChessBot-Battleground

   ```bash
   # Either install from source...
   git clone https://github.com/KeithG33/ChessBot-Battleground.git
   cd ChessBot-Battleground
   pip install -r requirements.txt
   pip install e .  

   # Or install via pip (TODO)
   pip install ChessBot-Battleground
   ```


**2. Download the Dataset:**  

The dataset is provided as a downloadable .zip file with each release. Either navigate to the github release page, or use the provided download tool:
```bash
chessbot download [<tag>="latest"] [--save-path <path>] [--dataset-name <filename>]

# Default: download latest release to cwd if pip installed, or ChessBot-Battleground/dataset if source installed
chessbot download

# Ex: download release v0.0.0 to output_dir
chessbot download v0.0.0 --output-dir /path/to/output_dir
```

By default, the latest release will be downloaded into the `ChessBot-Battleground/dataset/` directory, or the current working directory if the package has been pip installed.  


**2. Training**  
Check out the examples in [`ChessBot-Battleground/examples/`](examples/) or the real use-cases in [`ChessBot-Battleground/models/`](models/) for full setups. There are lots of configurations to use. If you want to do something more custom feel free write your own training loop, or even dataset.

And post your test scores! 





