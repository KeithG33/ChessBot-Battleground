from omegaconf import OmegaConf
from chessbot.train import config
from chessbot.train import ChessTrainer

if __name__ == "__main__":
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
