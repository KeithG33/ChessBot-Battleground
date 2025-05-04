import os
import shutil

from huggingface_hub import snapshot_download

from chessbot.common import setup_logger


LOGGER = setup_logger("chessbot.download")
REPO_ID = "KeithG33/ChessBot-Dataset"


def determine_save_path(user_path=None) -> tuple[str, bool]:
    """ Determine the appropriate save path based on installation type. """
    if user_path:
        if not user_path.endswith("ChessBot-Dataset"):
            user_path = os.path.join(user_path, "ChessBot-Dataset")
        return user_path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if "site-packages" in script_dir or "dist-packages" in script_dir:
        LOGGER.info("Detected pip install. Defaulting to current working directory")
        return os.getcwd()
    
    dataset_dir = os.path.abspath(os.path.join(script_dir, "../../dataset/ChessBot-Dataset"))
    if not os.path.exists(dataset_dir):
        LOGGER.info(f"Creating dataset directory at: {dataset_dir}")
        os.makedirs(dataset_dir, exist_ok=True)  
    
    return dataset_dir


def download(output_dir: str = None, keep_raw_data: bool = False) -> None:
    output_dir = determine_save_path(output_dir)

    try:
        snapshot_download(
            repo_id=REPO_ID, 
            local_dir=output_dir, 
            repo_type="dataset",
            ignore_patterns=None if keep_raw_data else ["*.zip"],
        )

    except Exception as e:
        LOGGER.error(f"Error downloading -- {e}")

    # Clean up cache directory
    cache_dir = os.path.join(output_dir, ".cache")
    if os.path.exists(cache_dir):
        LOGGER.info(f"Cleaning up cache directory at: {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)