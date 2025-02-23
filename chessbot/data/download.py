import os
import requests
import zipfile
import re

from chessbot.common import setup_logger


_logger = setup_logger("chessbot.download")


REPO_OWNER = "KeithG33"
REPO_NAME = "ChessBot-Battleground"
GITHUB_TAGS_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/tags"


def get_latest_tag():
    """Fetch the latest release tag from GitHub."""
    try:
        response = requests.get(GITHUB_TAGS_URL)
        response.raise_for_status()
        tags = response.json()
        if tags:
            return tags[0]["name"]  # Example: "v0.0.0-test"
        else:
            _logger.error("No tags found in the repository.")
            return None
    except requests.exceptions.RequestException as e:
        _logger.error(f"Failed to fetch latest tag: {e}")
        return None


def extract_version(tag):
    """Extract the numeric version from a tag (e.g., 'v0.0.0-test' -> '0.0.0')."""
    match = re.search(r"(\d+\.\d+\.\d+)", tag)
    return match.group(1) if match else tag  # Fallback to original if extraction fails


def determine_save_path(user_path=None) -> tuple[str, bool]:
    """Determine the appropriate save path based on installation type.
    
    Returns True or False to indicate source or not"""
    if user_path:
        return user_path, False  # User explicitly provided a save path

    # Check if pip installed (inside site-packages)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if "site-packages" in script_dir or "dist-packages" in script_dir:
        _logger.info("Detected pip install. Defaulting to current working directory")
        return os.path.join(os.getcwd()), False  # Default to cwd/dataset

    source_dataset_dir = os.path.abspath(os.path.join(script_dir, "../../dataset"))
    return source_dataset_dir, True


def download(tag, output_dir, dataset_name, keep_raw_data=False):
    """Download and extract the dataset from GitHub.
    
    For pip installs, the dataset will be downloaded to the current working directory.
    For source installs, the dataset will be downloaded and extracted in the 'dataset' directory, by default.
    """
    
    tag = tag or get_latest_tag()
    if not tag:
        _logger.error("Could not determine a valid release tag. Exiting.")
        return
    
    version = extract_version(tag)
    dataset_name = dataset_name or f"ChessBot-Dataset-{version}.zip"
    download_url = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/{tag}/{dataset_name}"

    output_dir, source_install = determine_save_path(output_dir)
  
    _logger.info(f"Downloading dataset from {download_url}")

    try:
        response = requests.get(download_url, stream=True)
        response.raise_for_status()  # Raise an error for failed requests

        os.makedirs(output_dir, exist_ok=True)
        zip_path = os.path.join(output_dir, dataset_name)

        with open(zip_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        _logger.info(f"Dataset downloaded successfully: {zip_path}")

        # Extract the dataset if using source install
        if source_install:    
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)

            # Remove raw data if not keeping it
            if not keep_raw_data:
                raw_data_dir = os.path.join(output_dir, f"raw-data-{version}")
                if os.path.exists(raw_data_dir):
                    os.rmdir(raw_data_dir)
            _logger.info(f"Dataset extracted successfully in: {output_dir}")

    except requests.exceptions.RequestException as e:
        _logger.error(f"Failed to download the dataset. Error: {e}")
