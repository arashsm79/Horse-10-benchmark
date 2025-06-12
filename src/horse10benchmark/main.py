import argparse
import sys
from pathlib import Path
import glob
import shutil
from datetime import datetime
import tarfile
import logging
import os
import requests
import pickle
import random

import numpy as np
import pandas as pd
from ruamel.yaml import YAML


from config import config

def set_seed(seed: int):
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Setup and run Horse-10 benchmark with DLC.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to download and store the benchmark data.")
    parser.add_argument("--assets_dir", type=str, default="assets", help="Directory to where the assets are stored.")
    args = parser.parse_args()
    if args.data_dir == 'data':
        args.data_dir = os.path.join(os.getcwd(), 'data')
    config['data_dir'] = args.data_dir
    if args.assets_dir == 'assets':
        args.assets_dir = os.path.join(os.getcwd(), 'assets')
    config['assets_dir'] = args.assets_dir
    project_dir = os.path.join(args.data_dir, "dlc_project")
    os.makedirs(project_dir, exist_ok=True)
    config['project_dir'] =  project_dir
    config['horse10_data_link'] = "https://huggingface.co/datasets/mwmathis/Horse-30/resolve/main/horse10.tar.xz"

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

    config_file_path = os.path.join(project_dir_path, 'config.yaml')

    if os.path.exists(config_file_path):
        logging.info(f"Project already exists at {project_dir_path}, skipping creation.")
        return

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(os.path.join(config['assets_dir'], 'config.yaml'), 'r') as f:
        config_file_contents = yaml.load(f)

    config_file_contents['project_path'] = project_dir_path
    config_file_contents['date'] = datetime.now().strftime("%b%d")

    with open(config_file_path, 'w') as f:
        yaml.dump(config_file_contents, f)
    
    # Create necessary directories
    labeled_data_dir_path = os.path.join(project_dir_path, 'labeled-data')
    os.makedirs(labeled_data_dir_path, exist_ok=True)
    dlc_models_dir_path = os.path.join(project_dir_path, 'dlc-models')
    os.makedirs(dlc_models_dir_path, exist_ok=True)
    training_dataset_dir_path = os.path.join(project_dir_path, 'training-datasets')
    os.makedirs(training_dataset_dir_path, exist_ok=True)
    videos_dir_path = os.path.join(project_dir_path, 'videos')
    os.makedirs(videos_dir_path, exist_ok=True)

    # Copy labeled data
    shutil.copytree(os.path.join(config['data_dir'], 'horse10', 'labeled-data'), labeled_data_dir_path, dirs_exist_ok=True)
    logging.info(f"Copied labeled data to {labeled_data_dir_path}")

    import deeplabcut as dlc
    dlc.check_labels(config_file_path)

    # Create videos out of each labeled images
    import cv2
    labeled_data_dir_path = Path(labeled_data_dir_path)
    labeled_data_dirs = [file.resolve() for file in labeled_data_dir_path.iterdir() if not file.name.endswith('_labeled')]
    labeled_data_dirs = sorted(labeled_data_dirs, key=lambda x: x.name)
    for labeled_data_dir in labeled_data_dirs:
        video_path = os.path.join(videos_dir_path, f"{labeled_data_dir.name}.mp4") # Change to .avi for compatibility
        images = glob.glob(os.path.join(str(labeled_data_dir), '*.png'))
        images.sort()
        if images:
            frame = cv2.imread(images[0])
            h, w, layers = frame.shape
            fourcc = cv2.VideoWriter.fourcc(*'MJPG') # Lossless
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h), True)
            for image in images:
                img = cv2.imread(image)
                out.write(img)
            out.release()
            logging.info(f"Created video {video_path} from labeled data {labeled_data_dir.name}")
        

def create_dataset_splits():
    config_file_path = os.path.join(config['project_dir'], 'config.yaml')
    project_dir_path = config['project_dir']

    shuffle_indices_path = glob.glob(os.path.join(project_dir_path, 'training-datasets', '**', 'shufflesIndices.pkl'), recursive=True)
    if shuffle_indices_path:
      logging.info("Shuffle indices already exist, skipping creation.")
      return

    collated_labels_h5_path = glob.glob(os.path.join(project_dir_path, 'training-datasets', '**', '*.h5'), recursive=True)
    if not collated_labels_h5_path:
        import deeplabcut as dlc
        # Dummy training dataset to get the indices
        dlc.create_training_dataset(config_file_path, Shuffles=[99], trainIndices=[[0]], testIndices=[[0]])
        collated_labels_h5_path = next(iter(glob.glob(os.path.join(project_dir_path, 'training-datasets', '**', '*.h5'), recursive=True)))
    else:
        collated_labels_h5_path = collated_labels_h5_path[0]
    
    collated_labels_h5 = pd.read_hdf(collated_labels_h5_path)
    collated_labels = ['/'.join(path) for path in collated_labels_h5.index.tolist()]

    # Load shuffles from assets folder
    shuffle_csv_paths = glob.glob(os.path.join(config['assets_dir'], '**', '*_shuffle*.csv'), recursive=True)
    if not shuffle_csv_paths or len(shuffle_csv_paths) != 3:
        raise Exception("All shuffle CSV files were not found in the assets folder.")
      
    shuffle_csvs = [pd.read_csv(path) for path in shuffle_csv_paths]

    assert len(shuffle_csvs) == 3, "There should be exactly 3 shuffle CSV files."

    shuffle_indices = []

    yaml = YAML()
    with open(config_file_path, 'r') as f:
        config_contents = yaml.load(f)
    
    trainingset_indices = config_contents['TrainingFraction']

    # Create DLC shuffles with different parameters for each of the 3 shuffles
    import deeplabcut as dlc
    idx = 1
    for i, shuffle_csv in enumerate(shuffle_csvs):
        train_inds = []
        test_inds = []
        ood_inds = []
        
        for j, row in shuffle_csv.iterrows():
            if pd.notna(row['trainIndices']):
              train_inds.append(collated_labels.index(row['trainIndices']))
            if pd.notna(row['testIndices_withinDomain']):
              test_inds.append(collated_labels.index(row['testIndices_withinDomain']))
            if pd.notna(row['testIndices_acrossDomain']):
              ood_inds.append(collated_labels.index(row['testIndices_acrossDomain']))

        assert len(train_inds) > 1300 and len(test_inds) > 1300 and len(ood_inds) > 5100
        all_train_inds = train_inds
        
        splits_fractions = [0.5, 0.05]
        net_types = ['rtmpose_x', 'resnet_50']
        for net_type in net_types:
            for split_fraction in splits_fractions:
                # Since for the original shuffles, 50% is not exactly 50% we will avoid the sampling
                if split_fraction == 0.5:
                    train_inds = all_train_inds
                else:
                    train_inds = random.sample(all_train_inds, round(split_fraction * len(all_train_inds+test_inds)))

                trainFractionWithinDomain = split_fraction
                test_inds_combined = test_inds + ood_inds  # Merge test and ood. We will manually separate them out from the evaluation_results.
                trainFraction = round(len(train_inds) * 1.0 / (len(train_inds) + len(test_inds_combined)), 2)
                shuffle_idx = idx
                shuffle_data = {
                    'shuffle_idx': shuffle_idx,
                    'net_type': net_type,
                    'train_fraction_within_domain': trainFractionWithinDomain,
                    'train_fraction': trainFraction,
                    'training_set_index': trainingset_indices.index(trainFraction),
                    'train_indices': train_inds,
                    'test_indices': test_inds,
                    'ood_indices': ood_inds
                }
                shuffle_indices.append(shuffle_data)
                logging.info(f"Creating shuffle {shuffle_idx} net_type: {net_type} with {len(train_inds)} Training Samples, trainFractionWithinDomain {trainFractionWithinDomain} and trainFraction {trainFraction}.")
                dlc.create_training_dataset(config_file_path, Shuffles=[shuffle_idx], trainIndices=[train_inds], testIndices=[test_inds_combined], net_type=net_type)
                idx += 1

    # Save the shuffle indices to a file
    shuffle_indices_path = Path(collated_labels_h5_path).parent / 'shufflesIndices.pkl'
    with open(shuffle_indices_path, 'wb') as f:
        pickle.dump(shuffle_indices, f)

    logging.info(f"Shuffle indices saved to {shuffle_indices_path}")

        
def train_dlc_models(modelprefix=""):
    config_file_path = os.path.join(config['project_dir'], 'config.yaml')
    project_dir_path = config['project_dir']

    # Load the shuffle indices
    shuffle_indices_path = glob.glob(os.path.join(project_dir_path, 'training-datasets', '**', 'shufflesIndices.pkl'), recursive=True)
    if not shuffle_indices_path:
        raise Exception("Shuffle indices file not found.")
    shuffle_indices_path = shuffle_indices_path[0]
    with open(shuffle_indices_path, 'rb') as f:
        shuffle_indices_data = pickle.load(f)

    from deeplabcut.utils import auxiliaryfunctions
    cfg = auxiliaryfunctions.read_config(config_file_path)

    # Train the models for each shuffle
    pytorch_config = {'runner.eval_interval': 100, "runner.snapshots.max_snapshots": 999} # Don't care about this since we are going to manually evaluate every snapshot afterwards
    import deeplabcut as dlc
    for shuffle_data in shuffle_indices_data:
        # Check if the training results already exist
        trained_model_dir_path = glob.glob(os.path.join(project_dir_path, 'dlc-models-pytorch', '**', f'*trainset{int(cfg["TrainingFraction"][int(shuffle_data["training_set_index"])]*100)}shuffle{shuffle_data["shuffle_idx"]}'), recursive=True)
        if trained_model_dir_path:
            logging.info(f"Trained model for shuffle {shuffle_data['shuffle_idx']} already exists, skipping training.")
            continue
        logging.info(f"Training model for shuffle {shuffle_data['shuffle_idx']} net_type: {shuffle_data['net_type']} with trainFractionWithinDomain {shuffle_data['train_fraction_within_domain']} and trainFraction {shuffle_data['train_fraction']}.")
        dlc.train_network(config_file_path, shuffle=shuffle_data['shuffle_idx'], trainingsetindex=shuffle_data['training_set_index'], save_epochs=40, max_snapshots_to_keep=999, pytorch_cfg_updates=pytorch_config)
    
def evaluate_dlc_models():
    config_file_path = os.path.join(config['project_dir'], 'config.yaml')
    project_dir_path = config['project_dir']

    # Load the shuffle indices
    shuffle_indices_path = glob.glob(os.path.join(project_dir_path, 'training-datasets', '**', 'shufflesIndices.pkl'), recursive=True)
    if not shuffle_indices_path:
        raise Exception("Shuffle indices file not found.")
    shuffle_indices_path = shuffle_indices_path[0]
    with open(shuffle_indices_path, 'rb') as f:
        shuffle_indices_data = pickle.load(f)

    from deeplabcut.utils import auxiliaryfunctions
    cfg = auxiliaryfunctions.read_config(config_file_path)

    # Evaluate the models for each shuffle
    import deeplabcut as dlc
    for shuffle_data in shuffle_indices_data:
        # Check if the evaluation results already exist
        shuffle_eval_dir_path = glob.glob(os.path.join(project_dir_path, 'evaluation-results-pytorch', '**', f'*trainset{int(cfg["TrainingFraction"][int(shuffle_data["training_set_index"])]*100)}shuffle{shuffle_data["shuffle_idx"]}'), recursive=True)
        if shuffle_eval_dir_path:
            logging.info(f"Evaluation results for shuffle {shuffle_data['shuffle_idx']} already exist, skipping evaluation.")
            continue
        logging.info(f"Evaluating model for shuffle {shuffle_data['shuffle_idx']} net_type: {shuffle_data['net_type']} with trainFractionWithinDomain {shuffle_data['train_fraction_within_domain']} and trainFraction {shuffle_data['train_fraction']}.")
        dlc.evaluate_network(config_file_path, Shuffles=[shuffle_data['shuffle_idx']], trainingsetindex=shuffle_data['training_set_index'], snapshotindex="all", per_keypoint_evaluation=True)

def plot_error_over_epoch():
    def convert_dlc_scores_to_numpy(score_data):
        # Get list of unique body parts
        body_parts = score_data.columns.get_level_values('bodyparts').unique().tolist()
        # Convert to numpy array with shape [frames, body_parts, xy]
        num_frames = len(score_data)
        keypoint_array = np.zeros((num_frames, len(body_parts), 2))
        scorer = score_data.columns.get_level_values(0)[0]
        # Fill the array with data
        for i, body_part in enumerate(body_parts):
            x_values = score_data.loc[:, (scorer, body_part, 'x')].values
            y_values = score_data.loc[:, (scorer, body_part, 'y')].values
            keypoint_array[:, i, 0] = x_values
            keypoint_array[:, i, 1] = y_values

        return keypoint_array

    config_file_path = os.path.join(config['project_dir'], 'config.yaml')
    project_dir_path = config['project_dir']

    # Load the shuffle indices
    shuffle_indices_path = glob.glob(os.path.join(project_dir_path, 'training-datasets', '**', 'shufflesIndices.pkl'), recursive=True)
    if not shuffle_indices_path:
        raise Exception("Shuffle indices file not found.")
    shuffle_indices_path = shuffle_indices_path[0]
    with open(shuffle_indices_path, 'rb') as f:
        shuffle_indices_data = pickle.load(f)

    # Load horsescale
    horse_scale_df = pd.read_hdf(os.path.join(config['assets_dir'], 'Horsescale.h5')) 

    # Load annotated data
    annotated_data_path = glob.glob(os.path.join(project_dir_path, 'training-datasets', '**', 'CollectedData_*.h5'), recursive=True)
    if not annotated_data_path:
        raise Exception("Annotated data file not found.")
    annotated_data_path = annotated_data_path[0]
    reference_data_df = pd.read_hdf(annotated_data_path)


    # Reorder horsescale to match the annotated data
    index_tuples = []
    scales = []
    for row_index, row in reference_data_df.iterrows():
        scale = horse_scale_df.loc['/'.join(row_index)]['scale']
        index_tuples.append(row_index)
        scales.append(scale)
    multi_index = pd.MultiIndex.from_tuples(index_tuples)
    horse_scale_df = pd.DataFrame(scales, index=multi_index, columns=['scale'])
    horse_scale_arr = np.array(horse_scale_df['scale'].values)

    from deeplabcut.utils import auxiliaryfunctions
    cfg = auxiliaryfunctions.read_config(config_file_path)

    import deeplabcut as dlc
    for shuffle_data in shuffle_indices_data:
        if shuffle_data['net_type'] == 'rtmpose_x':
            shuffle_eval_dir_path = glob.glob(os.path.join(project_dir_path, 'evaluation-results-pytorch', '**', f'*trainset{int(cfg["TrainingFraction"][int(shuffle_data["training_set_index"])]*100)}shuffle{shuffle_data["shuffle_idx"]}'), recursive=True)
            if not shuffle_eval_dir_path:
                raise Exception(f"Evaluation results for shuffle {shuffle_data['shuffle_idx']} not found.")
            shuffle_eval_dir_path = shuffle_eval_dir_path[0]

            # Get all the h5 files in the shuffle evaluation directory
            snapshot_h5_files_paths = glob.glob(os.path.join(shuffle_eval_dir_path, '*snapshot*.h5'))
            snapshot_h5_files_paths.sort(key=lambda x: int(x.split('_snapshot_')[-1].split('.')[0]))  # Sort by epoch number
            
            results = {
                "iid": [],
                "ood": [],
                "train": [],
                "epoch": [],
            }

            for snapshot_h5_file_path in snapshot_h5_files_paths:
                snapshot_h5_file = pd.read_hdf(snapshot_h5_file_path)

                # Tidy up the snapshot_h5_file
                predicted_data_df = snapshot_h5_file.loc[:, snapshot_h5_file.columns[snapshot_h5_file.columns.get_level_values(3) != 'likelihood']]
                predicted_data_df.columns = predicted_data_df.columns.droplevel('individuals')
                # Get the scorer name from reference_data
                reference_scorer = reference_data_df.columns.get_level_values(0)[0]
                # Rename the scorer in predicted_data to match reference_data
                predicted_data_df.columns = predicted_data_df.columns.set_levels([reference_scorer], level=0)

                predicted_arr = convert_dlc_scores_to_numpy(predicted_data_df)
                reference_arr = convert_dlc_scores_to_numpy(reference_data_df)

                # Calculate the error
                error_per_keypoint = np.linalg.norm(predicted_arr - reference_arr, axis=-1)
                error_per_keypoint_scaled = error_per_keypoint / horse_scale_arr[:, np.newaxis]

                results['iid'].append(error_per_keypoint_scaled[shuffle_data['test_indices'], :])
                results['ood'].append(error_per_keypoint_scaled[shuffle_data['ood_indices'], :])
                results['train'].append(error_per_keypoint_scaled[shuffle_data['train_indices'], :])
                results['epoch'].append(int(snapshot_h5_file_path.split('_snapshot_')[-1].split('.')[0]))

                # Save the results to a pickle file
                snapshot_eval_results_path = os.path.join(shuffle_eval_dir_path, f'snapshot_{int(snapshot_h5_file_path.split("_snapshot_")[-1].split(".")[0])}_results.pkl')


            # Calculate the mean error across keypoints for each frame and then average across frames
            mean_iid_error = [np.nanmean(np.nanmean(error, axis=1)) for error in results['iid']]
            mean_ood_error = [np.nanmean(np.nanmean(error, axis=1)) for error in results['ood']]
            mean_train_error = [np.nanmean(np.nanmean(error, axis=1)) for error in results['train']]

            # Create a plot
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(results['epoch'], mean_iid_error, 'bo-', label='Test (within domain)')
            plt.plot(results['epoch'], mean_ood_error, 'ro-', label='Test (out of domain)')
            plt.plot(results['epoch'], mean_train_error, 'go-', label='Train')

            plt.xlabel('Epoch')
            plt.ylabel('Normalized Error')
            plt.title(f'Mean Normalized Error vs. Epoch (Shuffle {shuffle_data["shuffle_idx"]})')
            plt.legend()
            plt.grid(True)

            # Save the plot
            plot_save_path = os.path.join(shuffle_eval_dir_path, f'error_vs_epoch_shuffle{shuffle_data["shuffle_idx"]}.png')
            plt.savefig(plot_save_path)
            plt.close()
            logging.info(f"Saved error vs. epoch plot to {plot_save_path}")

                
                



def main():
    set_seed(79)
    setup_logging()
    args = parse_args()
    # download_data(args.data_dir)
    # create_dlc_project()
    # create_dataset_splits()
    # train_dlc_models()
    # evaluate_dlc_models()
    plot_error_over_epoch()
    

if __name__ == "__main__":
    main()
