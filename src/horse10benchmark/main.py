import argparse
import shutil
from datetime import datetime
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
    project_dir = os.path.join(args.data_dir, "dlc_project")
    os.makedirs(project_dir, exist_ok=True)
    config['project_dir'] =  project_dir
    config['horse10_data_link'] = "https://huggingface.co/datasets/mwmathis/Horse-30/resolve/main/horse10.tar.xz"
    config['assets_dir'] = "assets"

    return args

def download_data(data_dir):
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

def create_dlc_project():
    project_dir_path = config['project_dir']

    if os.path.exists(os.path.join(project_dir_path, 'config.yaml')):
        logging.info(f"Project already exists at {project_dir_path}, skipping creation.")
        return

    config_file = f'''
# Project definitions (do not edit)
Task: horse10
scorer: Byron
date: {datetime.now().strftime("%b%d")}
multianimalproject: false
identity: false


# Project path (change when moving around)
project_path: {project_dir_path}


# Default DeepLabCut engine to use for shuffle creation (either pytorch or tensorflow)
engine: pytorch


# Annotation data set configuration (and individual video cropping parameters)
video_sets:
  /videos/BrownHorseintoshadow.mp4:
    crop: 0, 288, 0, 162
  /videos/Brownhorselight.mp4:
    crop: 0, 288, 0, 162
  /videos/Brownhorseoutofshadow.mp4:
    crop: 0, 288, 0, 162
  /videos/ChestnutHorseLight.mp4:
    crop: 0, 576, 0, 324
  /videos/Chestnuthorseongrass.mp4:
    crop: 0, 288, 0, 162
  /videos/GreyHorseLightandShadow.mp4:
    crop: 0, 288, 0, 162
  /videos/GreyHorseNoShadowBadLight.mp4:
    crop: 0, 288, 0, 162
  /videos/Sample1.mp4:
    crop: 0, 288, 0, 162
  /videos/Sample10.mp4:
    crop: 0, 288, 0, 162
  /videos/Sample11.mp4:
    crop: 0, 288, 0, 162
  /videos/Sample12.mp4:
    crop: 0, 288, 0, 162
  /videos/Sample13.mp4:
    crop: 0, 288, 0, 162
  /videos/Sample14.mp4:
    crop: 0, 288, 0, 162
  /videos/Sample15.mp4:
    crop: 0, 288, 0, 162
  /videos/Sample16.mp4:
    crop: 0, 288, 0, 162
  /videos/Sample17.mp4:
    crop: 0, 288, 0, 162
  /videos/Sample18.mp4:
    crop: 0, 288, 0, 162
  /videos/Sample19.mp4:
    crop: 0, 288, 0, 162
  /videos/Sample2.mp4:
    crop: 0, 288, 0, 162
  /videos/Sample20.mp4:
    crop: 0, 288, 0, 162
  /videos/Sample3.mp4:
    crop: 0, 288, 0, 162
  /videos/Sample4.mp4:
    crop: 0, 288, 0, 162
  /videos/Sample5.mp4:
    crop: 0, 288, 0, 162
  /videos/Sample6.mp4:
    crop: 0, 288, 0, 162
  /videos/Sample7.mp4:
    crop: 0, 288, 0, 162
  /videos/Sample8.mp4:
    crop: 0, 288, 0, 162
  /videos/Sample9.mp4:
    crop: 0, 288, 0, 162
  /videos/TwoHorsesinvideobothmoving.mp4:
    crop: 0, 288, 0, 162
  /videos/Twohorsesinvideoonemoving.mp4:
    crop: 0, 288, 0, 162
  /videos/BrownHorseinShadow.MP4:
    crop: 0, 1920, 0, 1080

bodyparts:
- Nose
- Eye
- Nearknee
- Nearfrontfetlock
- Nearfrontfoot
- Offknee
- Offfrontfetlock
- Offfrontfoot
- Shoulder
- Midshoulder
- Elbow
- Girth
- Wither
- Nearhindhock
- Nearhindfetlock
- Nearhindfoot
- Hip
- Stifle
- Offhindhock
- Offhindfetlock
- Offhindfoot
- Ischium


# Fraction of video to start/stop when extracting frames for labeling/refinement
start: 0
stop: 1
numframes2pick: 20


# Plotting configuration
skeleton: []
skeleton_color: black
pcutoff: 0.6
dotsize: 4
alphavalue: 0.7
colormap: jet


# Training,Evaluation and Analysis configuration
TrainingFraction:
- 0.5
iteration: 0
default_net_type: resnet_50
default_augmenter: default
snapshotindex: -1
detector_snapshotindex: -1
batch_size: 4
detector_batch_size: 1


# Cropping Parameters (for analysis and outlier frame detection)
cropping: false
#if cropping is true for analysis, then set the values here:
x1: 0
x2: 640
y1: 277
y2: 624


# Refinement configuration (parameters from annotation dataset configuration also relevant in this stage)
corner2move2:
- 50
- 50
move2corner: true


# Conversion tables to fine-tune SuperAnimal weights
SuperAnimalConversionTables:
    '''
    config_file_path = os.path.join(project_dir_path, 'config.yaml')
    with open(config_file_path, 'w') as f:
        f.write(config_file)
    
    # Create necessary directories
    labeled_data_dir_path = os.path.join(project_dir_path, 'labeled-data')
    dlc_models_dir_path = os.path.join(project_dir_path, 'dlc-models')
    os.makedirs(dlc_models_dir_path, exist_ok=True)
    training_dataset_dir_path = os.path.join(project_dir_path, 'training-datasets')
    os.makedirs(training_dataset_dir_path, exist_ok=True)
    videos_dir_path = os.path.join(project_dir_path, 'videos')
    os.makedirs(videos_dir_path, exist_ok=True)

    # Copy labeled data
    shutil.copytree(os.path.join(config['data_dir'], 'horse10', 'labeled-data'), project_dir_path, dirs_exist_ok=True)
    logging.info(f"Copied labeled data to {labeled_data_dir_path}")

def create_dataset_splits():
    pass

def main():
    setup_logging()
    args = parse_args()
    download_data(args.data_dir)
    create_dlc_project()
    create_dataset_splits()
    

if __name__ == "__main__":
    main()