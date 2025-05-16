# Neural Functional: Learning Function to Scalar Maps for Neural PDE Surrogates

## Requirements

To install requirements:
```setup
conda create -n "my_env" 
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia 
conda install lightning -c conda-forge
pip install wandb h5py einops scikit-learn tqdm scipy
```

## Datasets
Due to anonymity, datasets are not released currently, but will be provided when published.

## Training

Workflow for training a model:
```
- Setup environment
- Download a dataset 
- Make a log directory 
- Setup wandb
- Set paths to dataset, normalization stats, logging directory
```

Configs are organized in the configs folder with the paths: configs/[model_type]/[pde].yaml

### Baselines
To train a baseline model:
```
python train.py --config=configs/base/{adv/kdv/swe}.yaml
```

### Neural Functionals 
To train a Hamiltonian Neural Functional:
```
python train.py --config=configs/hamiltonian/{adv/kdv/swe}.yaml
```

## Examples
A jupyter notebook (examples.ipynb) for running toy examples is provided. Additionally, the notebook serves as a tool to undestand neural functionals and visualize some of their properties.