
<!-- Banner Start -->
<div align="center">
<img src="assets/chessbot-banner.png" style="width: 45%; height: auto;">  

# ChessBot Battleground

**Chess AI Training & Battleground Platform**

[**Getting Started (Installation)**](#getting-started) • [**Dataset**](#-dataset) • [**Training**](#-training) • [**Inference & Battling**](#-inference--battling) • [**Models**](#models)  

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FKeithG33%2FChessBot-Battleground&count_bg=%23A7AFB3&title_bg=%23170532&icon=&icon_color=%230019EE&title=views&edge_flat=false)](https://hits.seeyoufarm.com)

</div>
<!-- Banner End -->

##  Introduction

This repository contains a gigantic curated chess dataset meant for machine learning, along with the supporting code to train, infer, and display games. All you need to do is design a model, and you can take advantage of any of the available features. In total this library provides support for:

- **Dataset/Training:** PyTorch dataset and training code
- **Gym Environment:** Gym environment for inference and self-play
- **Dueling:** Functionality for best-of-N matches between saved models
- **MCTS:** Simple implementation to give your supervised models search capability (*with training coming soon*).
- **Visualization:** Watch your models in action as they play and adapt on the board.
- **Evaluation**: Compare performance on the dataset. Leaderboard coming soon?

See a model in action below:

<div align="center"  id="chess-battle-gif">
  <img src="assets/chessSGU-R8.1-selfplay.gif" style="width: 35%; height: auto;">
  <p><em>Self-play after a few days of training a 300M parameter network from scratch on an RTX 3090.</em></p>
</div>


## 📂 Dataset
*[Releases available for download here](https://github.com/KeithG33/ChessBot-Battleground/releases)*  

Currently the dataset contains approximately **700 million positions** in **PGN format**, split across 1000 files. Huge credits to the following main sources:

- Lumbra's Database (filtered 2600+)
- Lichess Puzzle Database
- Computer Chess: TCEC Database, CCRL
<!-- - (*coming soon*) Stockfish Data: position evaluation, puzzle solutions, best-move sequences   -->
  
      
The PyTorch `ChessDataset` is provided in `chessbot/data/dataset.py` to load the data for training, which has the following format:

```python
import chessbot.data.ChessDataset

pgn_files   = 'path/to/pgn_files' # File, directory, or list of files
num_threads = 8

dataset     = ChessDataset(pgn_files, num_threads=num_threads)
dataloader  = DataLoader(dataset, batch_size=bsz)
batch       = next(iter(dataloader))

# Check shapes
states  = batch[0]  # (B, 8, 8),
actions = batch[1]  # (B, 4672)
results = batch[2]  # (B,)
```

If you have your own pgn files and want to make your own dataset, the PyTorch dataset should work with those as well. The moves and results are loaded from each game in the PGN file
## 🧠 Training

A `ChessTrainer` class can be used to train ChessBot models. The trainer splits the data loading and training into rounds and epochs. Each round will sample a new subset of `cfg.dataset.size_train` files, and then perform epochs on this subset.   


**ChessTrainer**
- HuggingFace's `accelerate` for easy access to mixed precision, compilation, gradient clipping, gradient accumulation, etc.
- Warmup LR scheduling and decay (linear or cosine)
- Validation frequency in iterations
- Optional logging to wandb 


Here's a somewhat realistic example of using it:

```python
from chessbot.config import load_default_cfg()
from chessbot.train import ChessTrainer
from chessnet import YourChessNet

# Get default cfg and do some basic setup
cfg = load_default_cfg() # get default cfg

cfg.dataset.data_path = 'ChessBot-Battleground/dataset/'
cfg.dataset.size_train = 25 # num files to sample for train set
cfg.dataset.size_test = 5 # num files to sample for test set
cfg.train.batch_size = 128
cfg.train.lr = 0.001
cfg.train.output_dir = 'output/'

model = YourChessNet()

trainer = ChessTrainer(cfg, model)
trainer.train()
```

See [`chessbot/train/config.yaml`](chessbot/train/config.yaml) for a list and description of the available options, and the next section for a brief description of models.


## 🤖 Models

To take advantage of the training and inference code, models should subclass the `BaseChessModel` class. The expected format is:
1. **Input**: `(B, 1, 8, 8)` tensor for position
2. **Output**: a policy distribution of shape `(B, 4672)`, and expected value of shape `(B, 1)`.

A minimal example of writing a model:
```python
from chessbot.models import BaseChessModel, ModelRegistry

@ModelRegistry.register()
class SimpleChessNet(BaseChessModel):
  """ One layer backbone and one layer prediction heads """
    
    def __init__(self):
        super().__init__()
        
        # Mini backbone
        self.backbone = nn.Linear(64, 256)

        # Policy head
        self.policy_head = nn.Linear(256, self.action_dim)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Tanh()  # Between -1 and 1 for lose, draw, win
        )

    def forward(self, x):
      """ Input is tensor of shape (B,1,8,8) """
        x             = x.view(B, -1)
        features      = self.backbone(x)            # -> (B, 256)
        action_logits = self.policy_head(features)  # -> (B, 4672)
        board_val     = self.value_head(features)   # -> (B, 1)

        return action_logits, board_val
```

As long as your model has these inputs and outputs, then all the features of the library are available. If you need to break that input/output format, then you'll have to write your own training and inference code.

The model registry is a simple helper for the library to find and load your chess models from a path and name. If the decorator input is empty as in the example, `@ModelRegistry.register()`, then the model will automatically be registered with the class name `SimpleChessNet`. This helps find and load models for command line scripts.
## 🦾 Inference & Battling

Take your models to the battleground!  

The library depends on an **Adversarial Gym Environment** designed for two-player turn-based games, that can be used to visualize model inference. To visualize your models playing, you can check out the functions in [`chessbot.inference`](chessbot/inference/):

```python
from example_model.simple_chessnet import SimpleChessNet
from chessbot.inference import selfplay, play_match

# Run selfplay. Returns value in [-1,0,1] for white's outcome
model = SimpleChessNet()
outcome = selfplay(model, visualize=True)

# Play a match with two models, use MCTS
model1 = SimpleChessNet()
model2 = SimpleChessNet()
scores = play_match(model1, model2, best_of=11, search=True, visualize=True) # Returns (score1, score2)
```

Use the search flag to harness **Monte Carlo Tree Search (MCTS)** for search during inference. *MCTS training code coming soon!*

The [Chess Battle GIF](#chess-battle-gif) at the beginning is an example of visualizing the game with the Chess-env, and using MCTS for test-time powered inference. 

## 📈 Future Plans

- Clean and deduplicate the dataset
- Expand the dataset, add Stockfish generated data, get to an epic milestone of **1 billion positions**.  
- Release **MCTS training pipelines**.  
- Add enhanced tools for training and visualization and evaluation.
- Add a leaderboard


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

The dataset is provided as a downloadable .zip file with each release. Either navigate to the github release page, or use the chessbot cli tool:
```bash
# For list of options
chessbot download --help 

# Default: download latest release to cwd if pip installed, or ChessBot-Battleground/dataset if source installed
chessbot download

# Ex: download release v0.0.0 to output_dir
chessbot download v0.0.0 --output-dir /path/to/output_dir
```

By default, the latest release will be downloaded into the `ChessBot-Battleground/dataset/` directory, or the current working directory if the package has been pip installed.  


**3. Examples**  

1. There is a simple and complete example in [examples](examples/) to get you started. The directory contains an example model and notebooks for training and inference. Check out the `SimpleChessNet` for an example of the model interface; use `example_training.ipynb` to train the model; use `example_inference.ipynb` to either run inference with the base model, or with an MCTS wrapper for a test-time-powerup.


2. For actual models check out the [models](models/) directory.

3. Additionally, an `example_sf_datagen.ipynb` exists to show how one might add data to the dataset. Unfortunately stockfish is slow so this is a hopeful crumb that I leave for the crowd


#### 4. Leaderboard / Evaluation
Share your model's results on the test set. Compare your scores against the leaderboard. Once you've trained a model run the provided evaluate script to get your test set metrics.

```python
from chessbot.inference.evaluate import evaluate_model

# Load your model
model = ChessModel()

# Evaluate the model
batch_size = 3072
num_threads = 8
dataset_dir = 'path/to/dataset/'
evaluate_model(model, pgn_dir, batch_size, num_threads, device='cuda')
```
<!-- 
From the command line:

```bash
# For list of options
chessbot evaluate --help 

# Default: register 'my_model' from 'path/to/model_dir', load data from 'ChessBot-Battleground/dataset'
chessbot evaluate "my_model" --model-dir "path/to/model_dir/" 
```  -->





