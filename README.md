# Neural Functional: Learning Function to Scalar Maps for Neural PDE Surrogates [Paper](https://arxiv.org/abs/2505.13275)

## Requirements

To install requirements:
```setup
conda create -n "my_env" 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install lightning -c conda-forge
pip install wandb h5py einops scikit-learn tqdm scipy matplotlib
```

To run examples.ipynb:
```
pip install ipykernel
```

## Datasets
All datasets used are released on [HuggingFace](https://huggingface.co/datasets/ayz2/hamiltonian_pdes). 


## Examples
A jupyter notebook (examples.ipynb) for running examples involving neural functionals is provided. 

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
