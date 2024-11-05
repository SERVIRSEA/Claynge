"""
DataModule for the Chesapeake Bay dataset for segmentation tasks.

This implementation provides a structured way to handle the data loading and
preprocessing required for training and validating a segmentation model.

Dataset citation:
Robinson C, Hou L, Malkin K, Soobitsky R, Czawlytko J, Dilkina B, Jojic N.
Large Scale High-Resolution Land Cover Mapping with Multi-Resolution Data.
Proceedings of the 2019 Conference on Computer Vision and Pattern Recognition
(CVPR 2019).

Dataset URL: https://lila.science/datasets/chesapeakelandcover
"""

import re
from pathlib import Path

import lightning as L
import numpy as np
import torch
import yaml
from box import Box
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import rasterio


class changeData(Dataset):
    """
    Dataset class for Chesapeake Bay segmentation that handles before/after images in GeoTIFF format.

    Args:
        before_chip_dir (str): Directory containing the 'before' GeoTIFF image chips.
        after_chip_dir (str): Directory containing the 'after' GeoTIFF image chips.
        label_dir (str): Directory containing the labels in GeoTIFF format.
        metadata (Box): Metadata for normalization and other dataset-specific details.
        platform (str): Platform identifier used in metadata.
    """

    def __init__(self, before_chip_dir, after_chip_dir, label_dir, metadata, platform):
        self.before_chip_dir = Path(before_chip_dir)
        self.after_chip_dir = Path(after_chip_dir)
        self.label_dir = Path(label_dir)
        self.metadata = metadata  
        self.wavelengths = list(metadata[platform].bands.wavelength.values())
        
        self.transform = self.create_transforms(
            mean=list(metadata[platform].bands.mean.values()),
            std=list(metadata[platform].bands.std.values()),
        )

        # Load chip and label file names
        self.before_chips = sorted([chip_path.name for chip_path in self.before_chip_dir.glob("*.tif")])
        self.after_chips = sorted([chip_path.name for chip_path in self.after_chip_dir.glob("*.tif")])
        self.labels = sorted([chip_path.name for chip_path in self.label_dir.glob("*.tif")])

    def create_transforms(self, mean, std):
        """
        Create normalization transforms.

        Args:
            mean (list): Mean values for normalization.
            std (list): Standard deviation values for normalization.

        Returns:
            torchvision.transforms.Compose: A composition of transforms.
        """
        return v2.Compose([
            v2.Normalize(mean=mean, std=std),
        ])

    def load_geotiff(self, file_path):
        """
        Load GeoTIFF file using rasterio.

        Args:
            file_path (str): Path to the GeoTIFF file.

        Returns:
            np.ndarray: Image array.
        """
        with rasterio.open(file_path) as src:
            image = src.read()  # Reads the image as (Bands, Height, Width)
        return image

    def __len__(self):
        return len(self.before_chips)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing 'before' image, 'after' image, and label.
        """
        # Ensure that the before, after, and label chips match by using their common filename part
        

        before_chip_name = self.before_chips[idx]
        common_name = before_chip_name  # Use the name or extract the common part here if needed
                
        # Match the after and label chips based on the common part of the name
        after_chip_name = common_name  # Assuming after and before chips follow the same naming pattern
        label_name = common_name  # Assuming labels follow the same naming pattern
        
        before_chip_path = self.before_chip_dir / before_chip_name
        after_chip_path = self.after_chip_dir / after_chip_name
        label_path = self.label_dir / label_name

        # Load before, after, and label GeoTIFFs
        selected_band_indexes = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
        before_image = self.load_geotiff(before_chip_path)[selected_band_indexes, :, :].astype(np.float32)
        after_image = self.load_geotiff(after_chip_path)[selected_band_indexes, :, :].astype(np.float32)
        label_image = self.load_geotiff(label_path).astype(np.float32)

        # Apply the normalization transform to both before and after images
        before_image = self.transform(torch.from_numpy(before_image))
        after_image = self.transform(torch.from_numpy(after_image))

        
        wavelengths = torch.tensor(self.wavelengths)

        second_band = 1-label_image[0]   # Convert boolean to int for consistency
        label_image = np.stack((label_image[0], second_band), axis=0)


        # Create datacubes for both before and after images
        before_sample = {
            "pixels": before_image,
            "label": torch.from_numpy(label_image),
            "waves":wavelengths,
            "time": torch.zeros(4),  # Placeholder for time information
            "latlon": torch.zeros(4),  # Placeholder for lat/lon information
        }

        after_sample = {
            "pixels": after_image,
            "label": torch.from_numpy(label_image),
            "waves":wavelengths,
            "time": torch.zeros(4),  # Placeholder for time information
            "latlon": torch.zeros(4),  # Placeholder for lat/lon information
        }

        # Return the dictionary containing both before and after samples, along with the label
        return {
            "before": before_sample,
            "after": after_sample,
            "label": torch.from_numpy(label_image),  # Optionally include the label separately
            "filename":common_name
        }
            


class changeDataModule(L.LightningDataModule):
    """
    DataModule class for the Chesapeake Bay dataset with support for before/after images.

    Args:
        train_before_chip_dir (str): Directory containing 'before' training image chips.
        train_after_chip_dir (str): Directory containing 'after' training image chips.
        train_label_dir (str): Directory containing training labels.
        val_before_chip_dir (str): Directory containing 'before' validation image chips.
        val_after_chip_dir (str): Directory containing 'after' validation image chips.
        val_label_dir (str): Directory containing validation labels.
        metadata_path (str): Path to the metadata file.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
        platform (str): Platform identifier used in metadata.
    """

    def __init__(  # noqa: PLR0913
        self,
        train_before_chip_dir,
        train_after_chip_dir,
        train_label_dir,
        val_before_chip_dir,
        val_after_chip_dir,
        val_label_dir,
        metadata_path,
        batch_size,
        num_workers,
        platform,
    ):
        super().__init__()
        self.train_before_chip_dir = train_before_chip_dir
        self.train_after_chip_dir = train_after_chip_dir
        self.train_label_dir = train_label_dir
        self.val_before_chip_dir = val_before_chip_dir
        self.val_after_chip_dir = val_after_chip_dir
        self.val_label_dir = val_label_dir
        self.metadata = Box(yaml.safe_load(open(metadata_path)))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.platform = platform

    def setup(self, stage=None):
        """
        Setup datasets for training and validation.

        Args:
            stage (str): Stage identifier ('fit' or 'test').
        """

        self.trn_ds = changeData(  # Updated dataset class to handle before/after
                before_chip_dir=self.train_before_chip_dir,
                after_chip_dir=self.train_after_chip_dir,
                label_dir=self.train_label_dir,
                metadata=self.metadata,
                platform=self.platform,
            )

        self.val_ds = changeData(  # Updated dataset class to handle before/after
                before_chip_dir=self.val_before_chip_dir,
                after_chip_dir=self.val_after_chip_dir,
                label_dir=self.val_label_dir,
                metadata=self.metadata,
                platform=self.platform,
            )

    def train_dataloader(self):
        """
        Create DataLoader for training data.

        Returns:
            DataLoader: DataLoader for training dataset.
        """
        return DataLoader(
            self.trn_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """
        Create DataLoader for validation data.

        Returns:
            DataLoader: DataLoader for validation dataset.
        """
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

