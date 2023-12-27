from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from Dataset import IMDBDataset
import pytorch_lightning as pl
from torchvision import transforms
from Dataset import IMDBDataset
from torch.utils.data import DataLoader

class IMDBModule(pl.LightningDataModule):
    def __init__(self, data_dir, metadata_file, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.metadata_file = metadata_file
        self.transforms = transforms.Compose(transforms.ToTensor(),
                                            transforms.Resize((256, 256), antialias=True),
                                            lambda x: x / 255)

    def setup(self, stage):
        if stage == 'fit':
            self.imdb_dataset = IMDBDataset(self.data_dir, self.metadata_file, self.batch_size, self.transforms)
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.imdb_dataset, 
                          batch_size=self.batch_size,
                          shuffle=True)
        
