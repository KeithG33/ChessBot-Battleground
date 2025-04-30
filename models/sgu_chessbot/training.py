import os
import chessbot
# from sgu_chessbot import SpatialGatingChessBot

from chessbot.models import MODEL_REGISTRY
from chessbot.train import HFChessTrainer

cwd = os.getcwd()
dataset_path = os.path.join(cwd, 'dataset/ChessBot-Dataset-0.1.0/dataset-0.1.0')

cfg = chessbot.config.get_cfg()

cfg.dataset.data_path = dataset_path
cfg.dataset.size_train = 200
cfg.dataset.size_test = 10
cfg.dataset.num_processes = 20

cfg.train.rounds = 50
cfg.train.epochs = 1 
cfg.train.batch_size = 3072
cfg.train.lr = 0.0003
cfg.train.scheduler = 'linear'
cfg.train.min_lr = 0.0001
cfg.train.warmup_lr = 0.00003
cfg.train.warmup_iters = 1000
cfg.train.compile = True
cfg.train.amp = 'bf16'
cfg.train.validation_every = 20_000

cfg.train.checkpoint_dir = '/home/kage/chess_workspace/ChessBot-Battleground/models/sgu_chessbot/2025-02-19_18-51-experiment/checkpoint'
cfg.train.resume_from_checkpoint = True

cfg.logging.wandb = False
cfg.logging.wandb_run_id = '03zh0g0j'


if __name__ == '__main__':
    # model = SpatialGatingChessBot()
    model = MODEL_REGISTRY.load_model('sgu_chessbot')
    trainer = HFChessTrainer(cfg, model)
    trainer.train()