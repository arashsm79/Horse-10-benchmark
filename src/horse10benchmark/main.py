import argparse
import tarfile
import logging
import os
import requests

from config import config

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Setup and run Horse-10 benchmark with DLC.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to download and store the benchmark data.")
    args = parser.parse_args()
    config['data_dir'] = args.data_dir
    config['project_dir'] = os.path.join(args.data_dir, "dlc_project")
    config['horse10_data_link'] = "https://huggingface.co/datasets/mwmathis/Horse-30/resolve/main/horse10.tar.xz"
    config['assets_dir'] = "assets"

    return args

def download_data_if_not_exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    files = os.listdir(data_dir)
    if 'horse10' in files:
        logging.info("Horse-10 data already exists, skipping download.")
    else:
        if "horse10.tar.xz" not in files:
            logging.info("Horse-10 data not found, downloading...")
            filename = os.path.join(data_dir, "horse10.tar.xz")
            r = requests.get(config['horse10_data_link'], stream=True)
            if r.status_code == 200:
                with open(filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                logging.info(f"Downloaded data to {filename}")
            else:
                raise Exception(f"Failed to download data: {r.status_code}")
        logging.info("Unpacking data...")
        with tarfile.open(filename, 'r:xz') as tar:
            tar.extractall(path=data_dir)
        logging.info("Unpacking complete.")

def main():
    setup_logging()
    args = parse_args()
    download_data_if_not_exists(args.data_dir)
    

if __name__ == "__main__":
    main()