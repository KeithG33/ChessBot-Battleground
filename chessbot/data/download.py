import argparse
import os
import zipfile
import requests
import tempfile

def download_and_extract(url, save_path):
    """Download a ZIP file from a web URL and extract it to a local directory."""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_zip_path = os.path.join(temp_dir, 'dataset.zip')

        print("Downloading dataset...")
        response = requests.get(url, allow_redirects=True)
        with open(temp_zip_path, 'wb') as f:
            f.write(response.content)

        print("Extracting dataset...")
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(save_path)

    print(f"Dataset downloaded and extracted to {save_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_save_path = os.path.join(script_dir, "../../dataset/")
    
    parser = argparse.ArgumentParser(description="Chess Dataset Downloader")
    parser.add_argument("--url", type=str, default="https://github.com/username/repository/releases/download/v1.0/chess_dataset.zip",
                        help="URL of the dataset to download")
    parser.add_argument("--save-path", type=str, default=default_save_path, help="Path to save the downloaded dataset")
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    download_and_extract(args.url, args.save_path)

if __name__ == "__main__":
    main()
