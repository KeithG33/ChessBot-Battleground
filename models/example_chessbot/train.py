from omegaconf import OmegaConf
from chessbot.train import config
from chessbot.train import HFChessTrainer

if __name__ == "__main__":
    # Get default OmegaConf cfg
    cfg = config.get_cfg()

    # Alternatively, load some config overrides (or an entire config) from a file
    cfg_load = OmegaConf.load('models/example_chessbot/config.yaml')

    # Override cfg with cfg_load, and add any new keys
    cfg = OmegaConf.merge(cfg, cfg_load)

    cfg.train.epochs = 25 # num epochs on sampled dataset
    cfg.train.batch_size = 128
    cfg.train.lr = 0.001
    cfg.train.output_dir = 'models/example_chessbot/output/'
    cfg.dataset.num_workers = 8
    cfg.dataset.shuffle_buffer = 100_000

    # Option 1: Load model from registry
    cfg.model.name = "simple_chessbot"
    cfg.model.kwargs = {"hidden_dim": 512}
    trainer = HFChessTrainer(cfg, load_model_from_config=True)
    trainer.train()

    # Option 2: Load model from path
    from simple_chessbot import SimpleChessBot
    model = SimpleChessBot(hidden_dim=512)
    trainer = HFChessTrainer(cfg, model=model)
    trainer.train()
