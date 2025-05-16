import lightning as L
from torch.utils.data import DataLoader

class PDEDataModule(L.LightningDataModule):
    def __init__(self, 
                 dataconfig,) -> None:
        
        super().__init__()
        self.data_config = dataconfig
        self.dataset_config = dataconfig["dataset"]
        self.batch_size = dataconfig["batch_size"]
        self.num_workers = dataconfig["num_workers"]
        self.pde = self.dataset_config["pde"]
        self.pin_memory = True

        if self.pde == "rsw" or self.pde == "swe":
            from data.dataset_2D import PDEDataset2D
            self.train_dataset = PDEDataset2D(**dataconfig["dataset"],
                                              split="train",
                                              path=dataconfig["train_path"])
            self.val_dataset = PDEDataset2D(**dataconfig["dataset"],
                                            split="valid",
                                            path=dataconfig["valid_path"])
        elif self.pde == "kdv" or self.pde == "burgers":
            from data.dataset_1D import PDEDataset1D
            self.train_dataset = PDEDataset1D(**dataconfig["dataset"],
                                              split="train",
                                              path=dataconfig["train_path"])
            self.val_dataset = PDEDataset1D(**dataconfig["dataset"],
                                            split="valid",
                                            path=dataconfig["valid_path"])
        elif self.pde == "advection": # testing different resolution for training and validation
            from data.dataset_1D import PDEDataset1D
            self.train_dataset = PDEDataset1D(**dataconfig["dataset"],
                                              path=dataconfig["train_path"],
                                              split="train",
                                              resolution=dataconfig['resolution_train'])
            self.val_dataset = PDEDataset1D(**dataconfig["dataset"],
                                            path=dataconfig["valid_path"],
                                            split="valid",
                                            resolution=dataconfig['resolution_valid'])
        else:
            raise ValueError("PDE not found")

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass
        
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        # Eager imports to avoid specific dependencies that are not needed in most cases

        if stage == "fit":
            pass 

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            pass

        if stage == "predict":
            pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers, 
                          pin_memory=self.pin_memory,
                          persistent_workers=True)

    def val_dataloader(self, eval=False):
        shuffle = True if eval else False 
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=shuffle, 
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=True)

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        return None