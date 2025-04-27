import os
import zipfile


from huggingface_hub import hf_hub_download

from chessbot.common import setup_logger


LOGGER = setup_logger("chessbot.download")
REPO_ID = "KeithG33/ChessBot-Dataset"
VERSION_LINKS = {
    "latest": "ChessBot-dataset-0.1.0.zip",
    "0.1.0": "ChessBot-dataset-0.1.0.zip",
}


def determine_save_path(user_path=None) -> tuple[str, bool]:
    """
    Determine the appropriate save path based on installation type.
    Returns a tuple: (save_path, is_source_install)
    """
    if user_path:
        return user_path, False

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if "site-packages" in script_dir or "dist-packages" in script_dir:
        LOGGER.info("Detected pip install. Defaulting to current working directory")
        return os.getcwd(), False
    
    if not os.path.exists(os.path.join(script_dir, "../../dataset")):
        LOGGER.info(f"Creating dataset directory at: {os.path.join(script_dir, '../../dataset')}")
        os.makedirs(os.path.join(script_dir, "../../dataset"), exist_ok=True)  
    
    source_dataset_dir = os.path.abspath(os.path.join(script_dir, "../../dataset"))
    return source_dataset_dir, True


def download(version: str = None, output_dir: str = None, keep_raw_data: bool = False):
    version = version or "latest"

    if version not in VERSION_LINKS:
        LOGGER.error(f"Version {version} not found. Available versions: {', '.join(VERSION_LINKS.keys())}")
        return
    
    filename = VERSION_LINKS[version]
    output_dir, source_install = determine_save_path(output_dir)

    if os.path.exists(os.path.join(output_dir, filename)) or (
        os.path.exists(os.path.join(output_dir, os.path.splitext(filename)[0]))
    ):
        LOGGER.info(f"{filename} already exists in {output_dir}. Skipping download.")
        return
        
    LOGGER.info(f"Downloading {filename} to {output_dir}...")

    try:
        hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=output_dir, repo_type="dataset")

        if source_install: # unzip
            with zipfile.ZipFile(os.path.join(output_dir, filename), "r") as zip_ref:
                non_raw_files = [mem for mem in zip_ref.namelist() if f'dataset-{version}/' in mem]
                members = zip_ref.namelist() if keep_raw_data else non_raw_files
                zip_ref.extractall(output_dir, members=members)
            LOGGER.info(f"Unzipped {filename} to {output_dir}")
    except Exception as e:
        LOGGER.error(f"Error downloading {filename} -- {e}")