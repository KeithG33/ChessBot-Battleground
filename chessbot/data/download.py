import argparse
import os
import requests
import zipfile
import logging
import re


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


REPO_OWNER = "KeithG33"
REPO_NAME = "ChessBot-Battleground"
GITHUB_TAGS_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/tags"

# Get highest numbered ChessBot-Dataset-* directory
def get_latest_dataset_dir():
    source_dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset"))
    dataset_dirs = [d for d in os.listdir(source_dataset_dir) if d.startswith("ChessBot-dataset-")]
    if not dataset_dirs:
        return None
    return os.path.join(source_dataset_dir, sorted(dataset_dirs)[-1])

DEFAULT_DATASET_DIR = get_latest_dataset_dir()


def get_latest_tag():
    """Fetch the latest release tag from GitHub."""
    try:
        response = requests.get(GITHUB_TAGS_URL)
        response.raise_for_status()
        tags = response.json()
        if tags:
            return tags[0]["name"]  # Example: "v0.0.0-test"
        else:
            logging.error("No tags found in the repository.")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch latest tag: {e}")
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
        logging.info("Detected pip install. Defaulting to current working directory")
        return os.path.join(os.getcwd()), False  # Default to cwd/dataset

    source_dataset_dir = os.path.abspath(os.path.join(script_dir, "../../dataset"))
    return source_dataset_dir, True


def download(tag, output_dir, dataset_name):
    """Download and extract the dataset from GitHub."""
    
    tag = tag or get_latest_tag()
    if not tag:
        logging.error("Could not determine a valid release tag. Exiting.")
        return
    
    version = extract_version(tag)
    dataset_name = dataset_name or f"test-{version}.zip"
    output_dir, source_install = determine_save_path(output_dir)

    download_url = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/{tag}/{dataset_name}"
    logging.info(f"Downloading dataset from {download_url}")

    try:
        response = requests.get(download_url, stream=True)
        response.raise_for_status()  # Raise an error for failed requests

        os.makedirs(output_dir, exist_ok=True)
        zip_path = os.path.join(output_dir, dataset_name)

        with open(zip_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        logging.info(f"Dataset downloaded successfully: {zip_path}")

        # Extract the dataset if using source install
        if source_install:    
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)

            logging.info(f"Dataset extracted successfully in: {output_dir}")

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download the dataset. Error: {e}")
