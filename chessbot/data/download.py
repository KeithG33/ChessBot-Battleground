import os
import re
import zipfile
import gdown

from chessbot.common import setup_logger


LOGGER = setup_logger("chessbot.download")

VERSION_TO_DRIVE_URL = {
    "0.1.0": "https://drive.google.com/file/d/1ywfMXdTwd2xhSfCuQbyPg3OZuTSST_rD/view?usp=sharing",
}


def parse_version(ver_str):
    """Parse semantic version string into a tuple of integers."""
    major, minor, patch = map(int, ver_str.split('.'))
    return (major, minor, patch)


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


def get_version(version):
    """Return the provided version or the latest version (by semantic versioning)."""
    if version is not None:
        return version
    return max(VERSION_TO_DRIVE_URL.keys(), key=parse_version)


def download(version, output_dir, keep_raw_data=False):
    """Download and extract the dataset from Google Drive using gdown."""
    version = get_version(version)
    drive_url = VERSION_TO_DRIVE_URL.get(version)

    if not drive_url:
        LOGGER.error("Version URL not found for the version provided.")
        return

    # Extract the file id from the drive URL.
    m = re.search(r'/file/d/([^/]+)', drive_url)
    file_id = m.group(1)

    dataset_name = f"ChessBot-Dataset-{version}.zip"
    output_dir, source_install = determine_save_path(output_dir)
    zip_path = os.path.join(output_dir, dataset_name)

    try:
        if os.path.exists(zip_path):
            LOGGER.info(f"Dataset already exists at: {zip_path}")
        else:
            gdown.download(id=file_id, output=zip_path, quiet=False)
            LOGGER.info(f"Dataset downloaded successfully: {zip_path}")

        # Extract zip to output dir if source installed.
        if source_install:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                non_raw_members = [mem for mem in zip_ref.namelist() if f'dataset-{version}/' in mem]
                members = zip_ref.namelist() if keep_raw_data else non_raw_members
                zip_ref.extractall(output_dir, members=members)
            LOGGER.info(f"Dataset extracted successfully in: {output_dir}")

    except Exception as e:
        LOGGER.error(f"Failed to download the dataset using gdown. Error: {e}")
