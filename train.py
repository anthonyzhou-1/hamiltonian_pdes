# Default imports
import argparse
from datetime import datetime
import torch
import os 

# Custom imports
from common.utils import get_yaml, save_yaml
from common.callbacks import PlottingCallback
from lightning.pytorch.callbacks import LearningRateMonitor
from data.datamodule import PDEDataModule
from modules.train_module import TrainModule
from common.normalizer import Normalizer

# Lightning imports
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

def process_args(args, config):
    modelconfig = config['model']
    trainconfig = config['training']
    dataconfig = config['data']

    if len(args.devices) > 0:
        trainconfig["devices"] = [int(device) for device in args.devices]
    if args.mode is not None:
        modelconfig["mode"] = args.mode
    if args.model_name is not None:
        modelconfig["model_name"] = args.model_name
    if args.wandb_mode is not None:
        trainconfig["wandb_mode"] = args.wandb_mode
    if args.kernel_name is not None:
        modelconfig["nf"]["kernel_name"] = args.kernel_name
    if args.seed is not None:
        trainconfig["seed"] = int(args.seed)
    
    return config, modelconfig, trainconfig, dataconfig

def main(args):
    config=get_yaml(args.config)
    config, modelconfig, trainconfig, dataconfig = process_args(args, config)
    description = args.description

    seed = trainconfig["seed"]
    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")

    kernel_name = modelconfig.get('nf', {}).get('kernel_name', "")
    name = modelconfig["model_name"] + "_" + dataconfig["dataset"]["pde"] + "_" +  modelconfig["mode"] + "_" + kernel_name + "_" + description + str(seed) + "_" + now
    wandb_logger = WandbLogger(project=trainconfig["project"],
                               name=name,
                               mode=trainconfig["wandb_mode"])
    path = trainconfig["default_root_dir"] + name + "/"

    os.makedirs(path, exist_ok=True) 
    save_yaml(config, path + "config.yml")

    checkpoint_callback = ModelCheckpoint(
        monitor="rollout_loss",
        filename= "model_{epoch:02d}-{rollout_loss:.2f}",
        mode='min',
        dirpath=path,
        save_last=True,
        save_top_k=1
    )

    plotting_callback = PlottingCallback()
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    datamodule = PDEDataModule(dataconfig=dataconfig)
    normalizer = Normalizer(**dataconfig["normalizer"], dataset=datamodule.train_dataset) if "normalizer" in dataconfig else None

    model = TrainModule(modelconfig=modelconfig,
                        normalizer=normalizer)

    accumulate_grads = trainconfig["accumulate_grad_batches"] if "accumulate_grad_batches" in trainconfig else 1
    trainer = L.Trainer(devices = trainconfig["devices"],
                        accelerator = trainconfig["accelerator"],
                        check_val_every_n_epoch = trainconfig["check_val_every_n_epoch"],
                        max_epochs = trainconfig["max_epochs"],
                        default_root_dir = path,
                        callbacks=[checkpoint_callback, plotting_callback, lr_monitor],
                        logger=wandb_logger,
                        num_sanity_val_steps=0, 
                        accumulate_grad_batches=accumulate_grads,)
    
    if trainconfig["checkpoint"] is not None:
        trainer.fit(model=model,
                datamodule=datamodule,
                ckpt_path=trainconfig["checkpoint"])
    else:
        trainer.fit(model=model, 
                datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument("--config", default=None)
    parser.add_argument('--devices', nargs='+', help='<Required> Set flag', default=[])
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--mode', default=None)
    parser.add_argument('--wandb_mode', default=None)
    parser.add_argument('--description', default="")
    parser.add_argument('--kernel_name', default=None)  
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    main(args)