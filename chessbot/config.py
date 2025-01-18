from pathlib import Path
import attridict
import yaml

def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    return attridict(config)

def get_config():
    """ Return a default configuration object """
    # load config.yaml from this file's directory
    cfg_path = Path(__file__).parent / 'config.yaml' 
    return load_config(cfg_path)