from pathlib import Path
from omegaconf import OmegaConf


def get_cfg():
    """ Return a default configuration object """
    # load config.yaml from this file's directory
    cfg_path = Path(__file__).parent / 'config.yaml' 
    return OmegaConf.load(cfg_path)

