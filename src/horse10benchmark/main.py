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
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
from ruamel.yaml import YAML
from omegaconf import OmegaConf

def setup_config() -> OmegaConf:
    """
Load and merge configurations from YAML and command line arguments.
    
    Returns:
        OmegaConf: Configuration object with merged settings from default and CLI
"""
    # Load default config from the config.yaml file
    default_config = OmegaConf.load(os.path.join(os.path.dirname(__file__), 'config.yaml'))
    
    # Get command line arguments directly with OmegaConf
    cli_config = OmegaConf.from_cli()
    
    # Merge configs (command line takes precedence)
    config = OmegaConf.merge(default_config, cli_config)
    
    # Create project directory
    os.makedirs(config.project_dir, exist_ok=True)
    
    return config

def setup_logging() -> None:
    """Configure basic logging for the application with INFO level and timestamp format."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_data(config: OmegaConf) -> None:
    """
    Download and extract the Horse-10 dataset if not already present.
    
    Args:
        config: Configuration object containing data paths and URLs
        
    Raises:
        Exception: If download fails
    """
    data_dir = config['data_dir']
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

def create_dlc_project(config: OmegaConf) -> None:
    """
    Create a DeepLabCut project with Horse-10 dataset and generate videos from labeled frames.
    
    Args:
        config: Configuration object containing project settings and paths
    """
    import deeplabcut as dlc
    import cv2
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

    dlc.check_labels(config_file_path)

    # Create videos out of each labeled images
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
        

def create_dataset_splits(config: OmegaConf) -> None:
    """
    Create training/testing dataset splits based on predefined shuffle indices.
    
    Creates multiple shuffles with different train/test splits and network types
    for benchmarking purposes.
    
    Args:
        config: Configuration object containing project settings
        
    Raises:
        Exception: If shuffle CSV files are not found or invalid
    """
    import deeplabcut as dlc
    config_file_path = os.path.join(config['project_dir'], 'config.yaml')
    project_dir_path = config['project_dir']

    shuffle_indices_path = glob.glob(os.path.join(project_dir_path, 'training-datasets', '**', 'shufflesIndices.pkl'), recursive=True)
    if shuffle_indices_path:
      logging.info("Shuffle indices already exist, skipping creation.")
      return

    collated_labels_h5_path = glob.glob(os.path.join(project_dir_path, 'training-datasets', '**', '*.h5'), recursive=True)
    if not collated_labels_h5_path:
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
        
        splits_fractions = [0.05, 0.5]
        net_types = config['net_types']
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

def train_model(config_file_path: str, shuffle_data: Dict[str, Any], seed: int, save_epochs: int) -> str:
    """
    Train a single DeepLabCut model with specified parameters.
    
    Args:
        config_file_path: Path to the DeepLabCut project configuration file
        shuffle_data: Dictionary containing shuffle indices and training parameters
        seed: Random seed for reproducibility
        save_epochs: Number of epochs between model checkpoints
    
    Returns:
        Success or error message
    """
    import deeplabcut as dlc
    try:
        logging.info(f"Training model for shuffle {shuffle_data['shuffle_idx']} net_type: {shuffle_data['net_type']} "
                    f"with trainFractionWithinDomain {shuffle_data['train_fraction_within_domain']} and "
                    f"trainFraction {shuffle_data['train_fraction']}.")
        
        pytorch_config = {'runner.eval_interval': 100, "runner.snapshots.max_snapshots": 999, "train_settings.seed": seed}
        dlc.train_network(config_file_path, 
                         shuffle=shuffle_data['shuffle_idx'], 
                         trainingsetindex=shuffle_data['training_set_index'], 
                         save_epochs=save_epochs, 
                         max_snapshots_to_keep=999, 
                         pytorch_cfg_updates=pytorch_config)
        
        return f"Completed training for shuffle {shuffle_data['shuffle_idx']}"
    except Exception as e:
        return f"Error training model for shuffle {shuffle_data['shuffle_idx']}: {str(e)}"

def train_dlc_models(config: OmegaConf) -> None:
    """
    Train multiple DeepLabCut models in parallel according to configuration.
    
    Utilizes concurrent.futures to parallelize training of models with different
    network architectures and training fractions.
    
    Args:
        config: Configuration object containing project settings
    
    Raises:
        Exception: If shuffle indices file not found
    """
    from deeplabcut.utils import auxiliaryfunctions
    import concurrent.futures
    
    config_file_path = os.path.join(config['project_dir'], 'config.yaml')
    project_dir_path = config['project_dir']
    max_workers = config['max_workers']

    # Load the shuffle indices
    shuffle_indices_path = glob.glob(os.path.join(project_dir_path, 'training-datasets', '**', 'shufflesIndices.pkl'), recursive=True)
    if not shuffle_indices_path:
        raise Exception("Shuffle indices file not found.")
    shuffle_indices_path = shuffle_indices_path[0]
    with open(shuffle_indices_path, 'rb') as f:
        shuffle_indices_data = pickle.load(f)

    cfg = auxiliaryfunctions.read_config(config_file_path)

    # Find models that need training
    models_to_train = []
    for shuffle_data in shuffle_indices_data:
        trained_model_dir_path = glob.glob(os.path.join(project_dir_path, 'dlc-models-pytorch', '**', 
                            f'*trainset{int(cfg["TrainingFraction"][int(shuffle_data["training_set_index"])]*100)}shuffle{shuffle_data["shuffle_idx"]}'), 
                            recursive=True)
        if not trained_model_dir_path:
            models_to_train.append(shuffle_data)
        else:
            logging.info(f"Trained model for shuffle {shuffle_data['shuffle_idx']} already exists, skipping training.")
    
    if not models_to_train:
        logging.info("All models have already been trained. Nothing to do.")
        return

    # Use ProcessPoolExecutor to run training in parallel
    logging.info(f"Starting parallel training of {len(models_to_train)} models with {max_workers} workers")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_shuffle = {executor.submit(train_model, config_file_path, data, config['seed'], 40): data for data in models_to_train}
        
        for future in concurrent.futures.as_completed(future_to_shuffle):
            shuffle_data = future_to_shuffle[future]
            try:
                result = future.result()
                logging.info(result)
            except Exception as e:
                logging.error(f"Training failed for shuffle {shuffle_data['shuffle_idx']}: {str(e)}")

def evaluate_model(config_file_path: str, shuffle_data: Dict[str, Any]) -> str:
    """
    Evaluate a single DeepLabCut model with specified parameters.
    
    Args:
        config_file_path: Path to the DeepLabCut project configuration file
        shuffle_data: Dictionary containing shuffle indices and training parameters
    
    Returns:
        Success or error message
    """
    import deeplabcut as dlc
    try:
        logging.info(f"Evaluating model for shuffle {shuffle_data['shuffle_idx']} net_type: {shuffle_data['net_type']} "
                f"with trainFractionWithinDomain {shuffle_data['train_fraction_within_domain']} and "
                f"trainFraction {shuffle_data['train_fraction']}.")
        
        dlc.evaluate_network(config_file_path, Shuffles=[shuffle_data['shuffle_idx']], 
                            trainingsetindex=shuffle_data['training_set_index'], 
                            snapshotindex="all", per_keypoint_evaluation=True)
        
        return f"Completed evaluation for shuffle {shuffle_data['shuffle_idx']}"
    except Exception as e:
        return f"Error evaluating model for shuffle {shuffle_data['shuffle_idx']}: {str(e)}"
    
def evaluate_dlc_models(config: OmegaConf) -> None:
    """
    Evaluate multiple DeepLabCut models in parallel according to configuration.
    
    Utilizes concurrent.futures to parallelize evaluation of models with different
    network architectures and training fractions.
    
    Args:
        config: Configuration object containing project settings
    
    Raises:
        Exception: If shuffle indices file not found or evaluation fails
    """
    from deeplabcut.utils import auxiliaryfunctions
    import concurrent.futures
    
    config_file_path = os.path.join(config['project_dir'], 'config.yaml')
    project_dir_path = config['project_dir']
    max_workers = config['max_workers']

    # Load the shuffle indices
    shuffle_indices_path = glob.glob(os.path.join(project_dir_path, 'training-datasets', '**', 'shufflesIndices.pkl'), recursive=True)
    if not shuffle_indices_path:
        raise Exception("Shuffle indices file not found.")
    shuffle_indices_path = shuffle_indices_path[0]
    with open(shuffle_indices_path, 'rb') as f:
        shuffle_indices_data = pickle.load(f)

    cfg = auxiliaryfunctions.read_config(config_file_path)

    # Find models that need evaluation
    models_to_evaluate = []
    for shuffle_data in shuffle_indices_data:
        shuffle_eval_dir_path = glob.glob(os.path.join(project_dir_path, 'evaluation-results-pytorch', '**', 
                           f'*trainset{int(cfg["TrainingFraction"][int(shuffle_data["training_set_index"])]*100)}shuffle{shuffle_data["shuffle_idx"]}'), 
                           recursive=True)
        if not shuffle_eval_dir_path:
            models_to_evaluate.append(shuffle_data)
        else:
            logging.info(f"Evaluation results for shuffle {shuffle_data['shuffle_idx']} already exist, skipping evaluation.")
    
    if not models_to_evaluate:
        logging.info("All models have already been evaluated. Nothing to do.")
        return

    # Use ProcessPoolExecutor to run evaluations in parallel
    logging.info(f"Starting parallel evaluation of {len(models_to_evaluate)} models with {max_workers} workers")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_shuffle = {executor.submit(evaluate_model, config_file_path, data): data for data in models_to_evaluate}
        
        for future in concurrent.futures.as_completed(future_to_shuffle):
            shuffle_data = future_to_shuffle[future]
            try:
                result = future.result()
                logging.info(result)
            except Exception as e:
                logging.error(f"Evaluation failed for shuffle {shuffle_data['shuffle_idx']}: {str(e)}")
                raise e

def calculate_error_over_epochs(config: OmegaConf) -> None:
    """
    Calculate error metrics over training epochs for all trained models.
    
    For each model, calculates in-domain (IID), out-of-domain (OOD), and training
    errors normalized by horse scale.
    
    Args:
        config: Configuration object containing project settings
    
    Raises:
        Exception: If required files are not found
    """
    from deeplabcut.utils import auxiliaryfunctions

    def convert_dlc_scores_to_numpy(score_data: pd.DataFrame) -> np.ndarray:
        """
        Convert DeepLabCut score DataFrame to numpy array with shape [frames, body_parts, xy].
        
        Args:
            score_data: DataFrame containing DeepLabCut scores
            
        Returns:
            Numpy array with shape [frames, body_parts, xy]
        """
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

    cfg = auxiliaryfunctions.read_config(config_file_path)

    for shuffle_data in shuffle_indices_data:
        shuffle_eval_dir_path = glob.glob(os.path.join(project_dir_path, 'evaluation-results-pytorch', '**', f'*trainset{int(cfg["TrainingFraction"][int(shuffle_data["training_set_index"])]*100)}shuffle{shuffle_data["shuffle_idx"]}'), recursive=True)
        if not shuffle_eval_dir_path:
            logging.warning(f"Evaluation results for shuffle {shuffle_data['shuffle_idx']} not found. Skipping error calculation.")
            continue

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
        snapshot_h5_file_path = Path(snapshot_h5_file_path)
        error_over_epochs_path = snapshot_h5_file_path.stem.split('_snapshot_')[0] + '_error-epochs.pkl'
        error_over_epochs_path = snapshot_h5_file_path.parent / error_over_epochs_path
        with open(error_over_epochs_path, 'wb') as f:
            pickle.dump(results, f)

def plot_error_over_epochs(config: OmegaConf) -> None:
    """
    Generate plots showing error metrics over training epochs for all trained models.
    
    Creates plots for each network type and training fraction, showing IID, OOD,
    and training errors across epochs. Also generates a CSV summary of best results.
    
    Args:
        config: Configuration object containing project settings
    
    Raises:
        Exception: If shuffle indices file not found
    """
    from deeplabcut.utils import auxiliaryfunctions
    import matplotlib.pyplot as plt

    config_file_path = os.path.join(config['project_dir'], 'config.yaml')
    project_dir_path = config['project_dir']

    # Load the shuffle indices
    shuffle_indices_path = glob.glob(os.path.join(project_dir_path, 'training-datasets', '**', 'shufflesIndices.pkl'), recursive=True)
    if not shuffle_indices_path:
        raise Exception("Shuffle indices file not found.")
    shuffle_indices_path = shuffle_indices_path[0]
    with open(shuffle_indices_path, 'rb') as f:
        shuffle_indices_data = pickle.load(f)

    cfg = auxiliaryfunctions.read_config(config_file_path)

    results: Dict[str, Dict[float, List[Dict[str, Any]]]] = {}
    for shuffle_data in shuffle_indices_data:
        shuffle_eval_dir_path = glob.glob(os.path.join(project_dir_path, 'evaluation-results-pytorch', '**', f'*trainset{int(cfg["TrainingFraction"][int(shuffle_data["training_set_index"])]*100)}shuffle{shuffle_data["shuffle_idx"]}'), recursive=True)
        if not shuffle_eval_dir_path:
            logging.warning(f"Evaluation results for shuffle {shuffle_data['shuffle_idx']} not found. Skipping error calculation.")
            continue
        # Load error over epochs
        shuffle_eval_dir_path = shuffle_eval_dir_path[0]
        error_over_epochs_path = glob.glob(os.path.join(shuffle_eval_dir_path, '*_error-epochs.pkl'))
        if not error_over_epochs_path:
            logging.warning(f"Error over epochs for shuffle {shuffle_data['shuffle_idx']} not found. Skipping plotting.")
            continue
        error_over_epochs_path = error_over_epochs_path[0]
        with open(error_over_epochs_path, 'rb') as f:
            error_over_epochs = pickle.load(f)

        # Calculate the mean error across keypoints for each frame and then average across frames
        mean_iid_error = [np.nanmean(np.nanmean(error, axis=1)) for error in error_over_epochs['iid']]
        mean_ood_error = [np.nanmean(np.nanmean(error, axis=1)) for error in error_over_epochs['ood']]
        mean_train_error = [np.nanmean(np.nanmean(error, axis=1)) for error in error_over_epochs['train']]

        if shuffle_data['net_type'] not in results:
            results[shuffle_data['net_type']] = {}
        if shuffle_data['train_fraction_within_domain'] not in results[shuffle_data['net_type']]:
            results[shuffle_data['net_type']][shuffle_data['train_fraction_within_domain']] = []
        
        results[shuffle_data['net_type']][shuffle_data['train_fraction_within_domain']].append({
            'epoch': error_over_epochs['epoch'],
            'iid_error': mean_iid_error,
            'ood_error': mean_ood_error,
            'train_error': mean_train_error,
            'shuffle_idx': shuffle_data['shuffle_idx']
        })
    
    results_csv = {
        'net_type': [],
        'fraction': [],
        'iid_error': [],
        'ood_error': [],
        'train_error': [],
    }
    # Create plots for each network type and training fraction
    for net_type, train_fractions in results.items():
        for train_fraction, shuffle_results in train_fractions.items():
            plt.figure(figsize=(12, 8))
            
            # Plot individual shuffle results with low alpha
            for idx, result in enumerate(shuffle_results):
                plt.plot(result['epoch'], result['iid_error'], 'b-', alpha=0.3)
                plt.plot(result['epoch'], result['ood_error'], 'r-', alpha=0.3)
                plt.plot(result['epoch'], result['train_error'], 'g-', alpha=0.3)
            
            # Calculate mean values
            # First ensure all results have the same epochs
            min_epochs = min(len(result['epoch']) for result in shuffle_results)
            common_epochs = shuffle_results[0]['epoch'][:min_epochs]
            
            # Calculate mean errors across shuffles
            mean_iid_error = np.mean([result['iid_error'][:min_epochs] for result in shuffle_results], axis=0)
            mean_ood_error = np.mean([result['ood_error'][:min_epochs] for result in shuffle_results], axis=0)
            mean_train_error = np.mean([result['train_error'][:min_epochs] for result in shuffle_results], axis=0)

            results_csv['net_type'].append(net_type)
            results_csv['fraction'].append(train_fraction)
            results_csv['train_error'].append(np.max(mean_train_error))
            results_csv['iid_error'].append(np.max(mean_iid_error))
            results_csv['ood_error'].append(np.max(mean_ood_error))
            
            # Plot means with high alpha and thicker lines
            plt.plot(common_epochs, mean_iid_error, 'bo-', linewidth=2, alpha=1.0, label='Test (within domain)')
            plt.plot(common_epochs, mean_ood_error, 'ro-', linewidth=2, alpha=1.0, label='Test (out of domain)')
            plt.plot(common_epochs, mean_train_error, 'go-', linewidth=2, alpha=1.0, label='Train')
            
            plt.xlabel('Epoch')
            plt.ylabel('Normalized Error')
            plt.title(f'{net_type} with {train_fraction*100:.1f}% Training Data')
            plt.legend()
            plt.grid(True)
            
            # Save the plot
            os.makedirs(os.path.join(project_dir_path, 'bechmark-results'), exist_ok=True)
            plot_save_path = os.path.join(project_dir_path, 'bechmark-results', f'error_vs_epoch_{net_type}_{int(train_fraction*100)}pct.png')
            plt.savefig(plot_save_path)
            plt.close()
            logging.info(f"Saved error vs. epoch plot to {plot_save_path}")

    results_csv_df = pd.DataFrame(results_csv)
    results_csv_df.to_csv(os.path.join(project_dir_path, 'bechmark-results', 'best-results.csv'), index=False)
    print(results_csv_df)

def main() -> None:
    """
    Main function that runs the complete Horse-10 benchmark pipeline.
    
    Steps:
    1. Setup logging and configuration
    2. Download Horse-10 dataset
    3. Create DeepLabCut project
    4. Create dataset splits
    5. Train models
    6. Evaluate models
    7. Calculate error metrics
    8. Generate plots and results summary
    """
    setup_logging()
    cfg = setup_config()
    if cfg.pipeline.download_data:
        download_data(cfg)
    if cfg.pipeline.create_dlc_project:
        create_dlc_project(cfg)
    if cfg.pipeline.create_dataset_splits:
        create_dataset_splits(cfg)
    if cfg.pipeline.train_dlc_models:
        train_dlc_models(cfg)
    if cfg.pipeline.evaluate_dlc_models:
        evaluate_dlc_models(cfg)
    if cfg.pipeline.calculate_metrics:
        calculate_error_over_epochs(cfg)
    if cfg.pipeline.save_plot_metrics:
        plot_error_over_epochs(cfg)

if __name__ == "__main__":
    main()