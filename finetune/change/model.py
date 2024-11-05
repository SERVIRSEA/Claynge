"""
LightningModule for training and validating a segmentation model using the
Segmentor class.
"""
import sys

import lightning as L
import segmentation_models_pytorch as smp
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchmetrics.classification import F1Score, MulticlassJaccardIndex

from finetune.change.factory import Segmentor
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex


SIZE = 128



class changeSegmentor(L.LightningModule):
    """
    LightningModule for segmentation tasks, utilizing Clay Segmentor.

    Attributes:
        model (nn.Module): Clay Segmentor model.
        loss_fn (nn.Module): The loss function.
        iou (Metric): Intersection over Union metric.
        f1 (Metric): F1 Score metric.
        lr (float): Learning rate.
    """

    def __init__(  # # noqa: PLR0913
        self,
        num_classes,
        feature_maps,
        ckpt_path,
        lr,
        wd,
        b1,
        b2,
    ):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for checkpointing

        self.model = Segmentor(
            num_classes=num_classes,
            feature_maps=feature_maps,
            ckpt_path=ckpt_path,
        )

        self.loss_fn = torch.nn.BCEWithLogitsLoss()
       
        
        self.lr = lr
        self.wd = wd
        
        #self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.iou = BinaryJaccardIndex()
        self.f1 = BinaryF1Score()
        


    def forward(self, before_datacube, after_datacube):
        """
        Forward pass through the segmentation model.

        Args:
            datacube (dict): A dictionary containing the input datacube and
                meta information like time, latlon, gsd & wavelenths.

        Returns:
            torch.Tensor: The segmentation logits.
        """

        waves = torch.tensor([0.493, 0.56, 0.665, 0.704, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19])  
        gsd = torch.tensor(10.0)  # NAIP GSD
        before_datacube['waves'] = waves
        after_datacube['waves'] = waves
        
        before_datacube['gsd'] = gsd
        after_datacube['gsd'] = gsd
        

        return self.model(before_datacube, after_datacube)



    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.
        Returns:
            dict: A dictionary containing the optimizer and scheduler configuration.
        """
        optimizer = optim.AdamW(
            [
                param
                for name, param in self.model.named_parameters()
                if param.requires_grad
            ],
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=1000,
            T_mult=1,
            eta_min=self.hparams.lr * 100,
            last_epoch=-1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
    
    def shared_step(self, batch, batch_idx, phase):
        """
        Shared step for training and validation, using both before and after images.

        Args:
            batch (dict): A dictionary containing 'before', 'after', and 'label' data.
            batch_idx (int): The index of the batch.
            phase (str): The phase (train or val).

        Returns:
            torch.Tensor: The loss value.
        """
        before_datacube = batch["before"]
        after_datacube = batch["after"]
        labels = batch["label"]  # Ground truth segmentation labels

        # Forward pass
        outputs = self(before_datacube, after_datacube)

        loss = self.loss_fn(outputs, labels)               
        
        # Apply sigmoid to get probabilities for binary segmentation
        probs = torch.sigmoid(outputs)

        # Calculate metrics using logits
        binary_labels = (labels > 0.5).float()  # Threshold labels
        iou = self.iou((torch.sigmoid(outputs) > 0.5).float(), binary_labels)
        f1 = self.f1((torch.sigmoid(outputs) > 0.5).float(), binary_labels)

        # Log metrics
        self.log(
            f"{phase}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{phase}/iou",
            iou,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{phase}/f1",
            f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def predict_step(self, batch, batch_idx):
        """
        Override the predict_step to handle batch structure for predictions.

        Args:
            batch (dict): Batch data with 'before' and 'after' images.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Model predictions for the batch.
        """
        before_datacube = batch['before']
        after_datacube = batch['after']
        
        # Forward pass
        outputs = self(before_datacube, after_datacube)
        
        # Apply sigmoid for binary mask
        predictions = torch.sigmoid(outputs)
        
        return predictions

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.shared_step(batch, batch_idx, "val")
