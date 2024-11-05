import pytorch_lightning as pl
#from pytorch_lightning.utilities.cli import LightningCLI
from lightning.pytorch.cli import LightningCLI

import sys

from finetune.change.datamodule import ChesapeakeDataModule  # noqa: F401
from finetune.change.model import ChesapeakeSegmentor  # noqa: F401


class SegmentModel(pl.LightningModule):
    def __init__(self, num_classes, lr, wd, b1, b2):
        super(SegmentModel, self).__init__()
        self.model = ChesapeakeModel(num_classes)
        self.lr = lr
        self.wd = wd
        self.b1 = b1
        self.b2 = b2

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd, betas=(self.b1, self.b2))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

if __name__ == "__main__":
    cli = LightningCLI(SegmentModel, ChesapeakeDataModule)
