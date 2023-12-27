import pytorch_lightning as pl
from Generator import Generator
from Discriminator import Discriminator

class GAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
    
    def training_step(self, batch):
         # generate some noise
         pass