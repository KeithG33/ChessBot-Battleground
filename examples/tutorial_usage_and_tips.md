# ChessBot Tutorial: Usage Guide

This tutorial provides an overview of the ChessBot repository, and will guide you through an end-to-end example, including any additional information that might be useful.

All examples provided are complete, allowing you to follow along and run the code yourself. Reach out if you encounter any issues!

The example model and training script are in [model/example_chessbot](../models/example_chessbot/)



## 1. Setup & Installation

First install the [Adversarial Gym](https://github.com/OperationBeatMeChess/adversarial-gym) chess environment:
  ```bash
  pip install adversarial-gym
  ```
Then install ChessBot-Battleground. Cloning the repo is suggested so you can take advantage of some quality-of-life features:

```bash
# (Recommended) Install from source...
git clone https://github.com/KeithG33/ChessBot-Battleground.git
cd ChessBot-Battleground
pip install -r requirements.txt
pip install e .  

# Or install via pip 
pip install git+https://github.com/KeithG33/ChessBot-Battleground.git
```

As a quick verification, you should now be able to run `chessbot --help` and see the help message.

```bash
>> chessbot --help

 Usage: chessbot [OPTIONS] COMMAND [ARGS]...                                                                                                                                                                                         
                                                                                                                                                                                                                                     
 ChessBot CLI Tool                                                                                                                                                                                                                   
                                                                                                                                                                                                                                     
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                                                                                                                                           │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.                                                                                                                    │
│ --help                        Show this message and exit.                                                                                                                                                                         │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ evaluate   Evaluate a model. Pass additional positional arguments with --model-arg and keyword arguments as a JSON string via --model-kwargs.                                                                                     │
│ download   Download a dataset.                                                                                                                                                                                                    │
│ play       Play a game against the bot using a loaded model. Pass additional positional arguments with --model-arg and keyword arguments as a JSON string via --model-kwargs.                                                     │
│ train      Train a model using the provided configuration file and optional overrides.                                                                                                                                            │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
## 2. Download the Dataset

*[Datasets available for download here](https://drive.google.com/drive/folders/1RylJnVbJTNRVc8i_XN1lCE0nwX2g_oG9?usp=sharing)* 

It is recommended to have the dataset in `ChessBot-Battleground/dataset/` so you can automatically load the dataset. Download or use the chessbot tool:

```bash
# Download to cwd if pip installed, or ChessBot-Battleground/dataset if source installed.
chessbot download
```

If source installed the dataset will be automatically extracted insied the dataset folder, ready for training. Otherwise a `Chessbot-Dataset-{version}.zip` will be in your current working directory.


## 3. Creating A ChessBot

Recall from the main [README](../README.md) that models have the format,

1. **Input**: `(B, 1, 8, 8)` tensor for position
2. **Output**: a policy distribution of shape `(B, 4672)`, and expected value of shape `(B, 1)`.

Here's a full implementation of a simple model, with the model registered as `"simple_chessbot"` and placed inside [models/example_chessbot](../models/example_chessbot/simple_chessbot.py):

```python
@ModelRegistry.register("simple_chessbot")
class SimpleChessBot(BaseChessBot):  
    def __init__(self):
        super().__init__()    

        # Simple mlp backbone
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 256),  # Flatten the 8x8 board and process
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(64, self.action_dim)
        ) 

        # Value head
        self.value_head  = nn.Sequential(
            nn.Linear(64, 1),
            nn.Tanh()  # Output a value between -1 and 1
        )

    def forward(self, x):
        features      = self.layers(x)
        action_logits = self.policy_head(features)
        board_val     = self.value_head(features)
        return action_logits, board_val
```

Now we can automatically load this model for training or inference by setting `cfg.model.name = "simple_chessbot"`, and with `cfg.model.args` or `cfg.model.kwargs` as needed.


## 4. Training Your ChessBot

#### Python

Training in python means you need to create a config by loading it or creating one at runtime. Or a combination of these solutions. Here we will show everything. The below code is [train.py](../models/example_chessbot/train.py)

```python
from omegaconf import OmegaConf
from chessbot.train import config
from chessbot.train import ChessTrainer

# Get default OmegaConf cfg
cfg = config.get_cfg()

# Alternatively, load some config overrides (or an entire config) from a file
cfg_load = OmegaConf.load('models/example_chessbot/config.yaml')

# Override cfg with cfg_load, and add any new keys
cfg = OmegaConf.merge(cfg, cfg_load)

cfg.train.rounds = 1 # num times to sample a dataset
cfg.train.epochs = 25 # num epochs on sampled dataset
cfg.train.batch_size = 128
cfg.train.lr = 0.001
cfg.train.output_dir = 'models/example_chessbot/output/'

cfg.dataset.size_train = 25 # num files to sample for train set
cfg.dataset.size_test = 5 # num files to sample for test set

# Option 1: Load model from registry
cfg.model.name = "simple_chessbot"
cfg.model.kwargs = {"hidden_dim": 512}
trainer = ChessTrainer(cfg, load_model_from_config=True)
trainer.train()

# Option 2: Load model from path
from simple_chessbot import SimpleChessBot
model = SimpleChessBot(hidden_dim=512)
trainer = ChessTrainer(cfg, model=model)
trainer.train()
```

The training output directory will store the config used and the best model, latest model, and the complete training state for resuming from checkpoint in a `checkpoint/` directory. 

To resume from checkpoint set `cfg.train.checkpoint_dir` to this `checkpoint/` directory. And to reuse the same training output directory set `resume_from_checkpoint=True`.
```python
# Use training state from previous train checkpoint (weights, optimizer, scheduler) 
cfg.train.checkpoint_dir = 'previous/train_dir/checkpoint/'

# Reuse previous train directory
cfg.train.resume_from_checkpoint = True

# Train ...
```

#### CLI
We can also use the command-line or a bash script to start training. See [train.sh](../models/example_chessbot/train.sh) for an example:


```bash
# Train from config, and any overrides in command
chessbot train models/example_chessbot/config.yaml \
              -o model.name=simple_chessbot \
              -o train.epochs=10 \
              -o train.lr=0.001 \
              -o train.batch_size=64 \
              -o dataset.size_train=2 \
              -o dataset.size_test=1 \
```

Loading from config with CLI functions the same as with python. Use `cfg.model.name`, and `cfg.model.path` if it is not inside the `models/` directory. 

> **Note:** `cfg.dataset.data_path` is not used here, which means the `DEFAULT_DATASET_DIR` is used. Set this to another dataset path if needed.

## 5. Evaluating Your ChessBot

One warning is that the test set is quite large and evaluation may take some time depending on model and hardware.
```python
from chessbot.inference import evaluate_model
from chessbot.common import DEFAULT_MODEL_DIR
from simple_chessbot import SimpleChessBot

model = SimpleChessBot(hidden_dim=512)
evaluate_model(
    model,
    dataset_dir=DEFAULT_MODEL_DIR,
    batch_size=64,
    num_processes=4,
    device="cuda",
    num_chunks=None,
)
```

Or if your model is registered as "your_chessbot", using the `chessbot` cli tool:
```bash
# For options and help:
chessbot evaluate --help

chessbot evaluate "your_chessbot" \ 
                  --model-dir path/to/dir \
                  --model-weights path/to/weights.pt \
                  --data-dir path/to/dataset \
                  --batch-sz 3072 \
                  --num-threads 8 \
```


## 6. Using Your ChessBot for Play

Instructions for integrating your trained ChessBot into real gameplay scenarios or platforms.

```python
# Gameplay integration example
```

## 7. Advanced Configuration

Explore advanced customization options for hyperparameters, architectures, and optimization.

```python
# Advanced configuration examples
```

## 8. Troubleshooting & FAQ

Answers to common issues and frequently asked questions.

```bash
# Common troubleshooting commands
```

---

## Contributing

We welcome contributions! Please follow the [contributing guidelines](CONTRIBUTING.md) to suggest improvements or new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Happy training, and enjoy your ChessBot!

