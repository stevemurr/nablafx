import torch
import lightning as pl
from typing import List, Union, Optional

from .datasets import PluginDataset, ParametricPluginDataset


# -----------------------------------------------------------------------------
# Datamodule for non-parametric and parametric datasets
# -----------------------------------------------------------------------------


class DryWetFilesPluginDataModule(pl.LightningDataModule):
    """Data module for pre-rendered audio examples from VST plugin.
    Separate dirs for dry and wet audio files.
    Assumes train and validation files are in the same directory (trainval).
    Assumes test files are in a separate directory (test).
    If trainval and test directories are the same, the same files will be used for training, validation and testing.

    Args:
        root_dir_dry (str): Path to directory containing dry audio files for training and validation or testing.
        root_dir_wet (str): Path to directory containing wet audio files for training and validation or testing.
        param_idxs_to_use (List): List of indices of parameters to use as conditioning.
        data_to_use (float): Fraction of data to use.
        trainval_split (float): Fraction of data to use for training.
        sample_length (int): Length of audio samples.
        sample_rate (int): Sample rate of audio files.
        preload (bool): Preload audio files into memory.
        batch_size (int): Size of batches.
        num_workers (int): Number of workers for data loading.
    """

    def __init__(
        self,
        root_dir_dry: str,
        root_dir_wet: str,
        params_idxs_to_use: Optional[Union[List[int], None]] = None,
        data_to_use: float = 1.0,
        trainval_split: float = 0.8,
        sample_length: int = -1,
        sample_rate: int = 48000,
        preload: bool = False,
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.root_dir_dry = root_dir_dry
        self.root_dir_wet = root_dir_wet
        self.params_idxs_to_use = params_idxs_to_use
        self.data_to_use = data_to_use
        self.trainval_split = trainval_split
        self.sample_length = sample_length
        self.sample_rate = sample_rate
        self.preload = preload
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage == "validate":
            if self.params_idxs_to_use is None:
                self.trainval_dataset = PluginDataset(
                    root_dir_dry=self.root_dir_dry,
                    root_dir_wet=self.root_dir_wet,
                    data_to_use=self.data_to_use,
                    sample_length=self.sample_length,
                    sample_rate=self.sample_rate,
                    preload=self.preload,
                )
            else:
                self.trainval_dataset = ParametricPluginDataset(
                    root_dir_dry=self.root_dir_dry,
                    root_dir_wet=self.root_dir_wet,
                    params_idxs_to_use=self.params_idxs_to_use,
                    data_to_use=self.data_to_use,
                    sample_length=self.sample_length,
                    sample_rate=self.sample_rate,
                    preload=self.preload,
                )

            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.trainval_dataset, [self.trainval_split, 1 - self.trainval_split]
            )

            print("Train and Validation Datasets:")
            self.trainval_dataset.print()
            print()

        if stage == "test":
            if self.params_idxs_to_use is None:
                self.test_dataset = PluginDataset(
                    root_dir_dry=self.root_dir_dry,
                    root_dir_wet=self.root_dir_wet,
                    data_to_use=self.data_to_use,
                    sample_length=self.sample_length,
                    sample_rate=self.sample_rate,
                    preload=self.preload,
                )
            else:
                self.test_dataset = ParametricPluginDataset(
                    root_dir_dry=self.root_dir_dry,
                    root_dir_wet=self.root_dir_wet,
                    params_idxs_to_use=self.params_idxs_to_use,
                    data_to_use=self.data_to_use,
                    sample_length=self.sample_length,
                    sample_rate=self.sample_rate,
                    preload=self.preload,
                )

            print("Test Dataset:")
            self.test_dataset.print()
            print()

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
