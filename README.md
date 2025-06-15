# Horse-10 Benchmark

A benchmark for evaluating pose estimation models on horse pose data, focusing on in-domain and out-of-domain generalization performance.

This repository provides an example of using this benchmark with [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut).

## Overview

Pose estimation is an important tool for measuring behavior, and thus widely used in technology, medicine and biology. Due to innovations in both deep learning algorithms and large-scale datasets pose estimation on humans has gotten very powerful. However, typical human pose estimation benchmarks, such as MPII pose and COCO, contain many different individuals (>10K) in different contexts, but only very few example postures per individual. In real world application of pose estimation, users want to estimate the location of user-defined bodyparts by only labeling a few hundred frames on a small subset of individuals, yet want this to generalize to new individuals. Thus, one naturally asks the following question: Assume you have trained an algorithm that performs with high accuracy on a given (individual) animal for the whole repertoire of movement - how well will it generalize to different individuals that have slightly or a dramatically different appearance? Unlike in common human pose estimation benchmarks here the setting is that datasets have many (annotated) poses per individual (>200) but only few individuals (1-25).
To allow the field to tackle this challenge, we developed a novel benchmark, called Horse-10, comprising 30 diverse Thoroughbred horses, for which 22 body parts were labeled by an expert in 8,114 frames. Horses have various coat colors and the “in-the-wild” aspect of the collected data at various Thoroughbred yearling sales and farms added additional complexity.


## Usage

The data is available on [HuggingFace](https://huggingface.co/datasets/mwmathis/Horse-30) (this code downloads it automatically).

The `assets` folder contains the files for the benchmark:
- **assets/config.yaml**: Base configuration for the DLC project
- **assets/TrainTestInfo_shuffle\*.csv**: Files defining the train/test/out-of-domain splits
- **assets/Horsescale.h5**: Contains normalization factors for different horse sizes. We normalize to the nose-eye distance for each horse.


This repository contains code to:
- Download the Horse-10 dataset
- Set up a DeepLabCut project
- Create standardized dataset splits
- Train models with different architectures and dataset fractions
- Evaluate performance across epochs
- Generate comparison plots


To run the example benchmark, first make sure you have [uv](https://github.com/astral-sh/uv) installed.

The project uses [OmegaConf](https://omegaconf.readthedocs.io) to manage its configuration. Use the config file located at `src/horse10benchmark/config.yaml` to specify which network types and training fractions you would like to benchmark. You can override the configuration with command line arguments as well.


```shell
git clone https://github.com/yourusername/Horse-10-benchmark.git

cd Horse-10-benchmark

uv run horse10benchmark
```

With command line arguments:
```shell
uv run horse10benchmark data_dir=/path/to/data/dir net_types=["rtmpose_x", "resnet_50"]
```

The result is available by default under `<data-dir>/dlc_project/benchmark-results`.

### Docker
You can also use the provided Dockerfile to build an image and run the benchmark in docker.

```shell
docker build -t horse10benchmark .

docker run -e PWD="/app" -v /host/path/to/data:/data -it horse10benchmark data_dir=/data net_types=["resnet_50"]
```

### DLC Version
You can specify the DLC version or commit that you would like to use in `pyproject.toml`. See the [uv docs](https://docs.astral.sh/uv/concepts/projects/dependencies/#git).


## References

For more details check out the paper: [Pretraining boosts out-of-domain robustness for pose estimation](https://openaccess.thecvf.com/content/WACV2021/html/Mathis_Pretraining_Boosts_Out-of-Domain_Robustness_for_Pose_Estimation_WACV_2021_paper.html) published in the IEEE Winter Conference on Applications of Computer Vision 2021.