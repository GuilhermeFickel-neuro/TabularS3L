from typing import Union
from torch.utils.data import Dataset, DataLoader, SequentialSampler, WeightedRandomSampler, RandomSampler

from pytorch_lightning import LightningDataModule


class TS3LDataModule(LightningDataModule):
    """The pytorch lightning datamodule for TabularS3L
    """

    def __init__(self, train_ds: Dataset, val_ds: Dataset, batch_size: int, train_sampler: str, train_collate_fn=None, valid_collate_fn=None, n_jobs: int = 32, drop_last: bool = False, is_regression: bool = False, prefetch_factor: int = 4, persistent_workers: bool = True, pin_memory: bool = True):
        """Initialize the datamodule

        Args:
            train_ds (Dataset): The training dataset.
            val_ds (Dataset): The validation dataset.
            batch_size (int): The batch size of the dataset.
            train_sampler (str): The training sampler for training dataset. It must be one of ["seq", "weighted", "random"].
            train_collate_fn (object): Collate function for train dataloader.
            valid_collate_fn (object): Collate function for validation dataloader.
            n_jobs (int, optional): The number of the cpu core to use. Defaults to 32.
            drop_last (bool, optional): The flag to drop the last batch or not. Defaults to False.
            is_regression (bool, optional): The flag that determines whether the datamodule is for regression task or not. Defaults to False.
            prefetch_factor (int, optional): Number of batches to prefetch. Defaults to 4.
            persistent_workers (bool, optional): Whether to keep worker processes alive. Defaults to True.
            pin_memory (bool, optional): Whether to pin memory for faster GPU transfer. Defaults to True.
        """
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds

        self.batch_size = batch_size

        assert train_sampler in [
            "seq", "weighted", "random"], 'The train_sampler must be one of the following: "seq", "weighted", or "random".'
        self.train_sampler = train_sampler

        self.train_collate_fn = train_collate_fn
        self.valid_collate_fn = valid_collate_fn

        self.n_jobs = n_jobs
        self.is_regression = is_regression
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers and n_jobs > 0  # Only use persistent workers if using multiple workers
        self.pin_memory = pin_memory

    def setup(self, stage: str):
        """Setup the datamodule for pytorch lightning module of TabularS3L.

        Use a weighted random sampler for the second phase step of the classification task, otherwise use a random sampler.

        Args:
            stage (str): Only for compatibility, not used
        """
        if self.train_sampler == "seq":
            sampler = SequentialSampler(self.train_ds)  # type: ignore
            shuffle = True
        elif self.train_sampler == "weighted":
            sampler = WeightedRandomSampler(
                self.train_ds.weights, num_samples=len(self.train_ds))  # type: ignore
            shuffle = False
        elif self.train_sampler == "random":
            sampler = RandomSampler(
                self.train_ds, num_samples=len(self.train_ds))  # type: ignore
            shuffle = False

        # Optimized training dataloader
        dataloader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': shuffle,
            'sampler': sampler,
            'num_workers': self.n_jobs,
            'drop_last': self.drop_last,
            'collate_fn': self.train_collate_fn,
            'pin_memory': self.pin_memory,
        }
        
        # Add prefetch_factor and persistent_workers only if using multiple workers
        if self.n_jobs > 0:
            dataloader_kwargs['prefetch_factor'] = self.prefetch_factor
            dataloader_kwargs['persistent_workers'] = self.persistent_workers

        self.train_dl = DataLoader(self.train_ds, **dataloader_kwargs)
        
        # Optimized validation dataloader
        val_dataloader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'sampler': SequentialSampler(self.val_ds),
            'num_workers': self.n_jobs,
            'drop_last': False,
            'collate_fn': self.valid_collate_fn,
            'pin_memory': self.pin_memory,
        }
        
        # Add prefetch_factor and persistent_workers only if using multiple workers
        if self.n_jobs > 0:
            val_dataloader_kwargs['prefetch_factor'] = self.prefetch_factor
            val_dataloader_kwargs['persistent_workers'] = self.persistent_workers

        self.val_dl = DataLoader(self.val_ds, **val_dataloader_kwargs)

    def train_dataloader(self):
        """Return the training dataloader.
        """
        return self.train_dl

    def val_dataloader(self):
        """Return the validation dataloader.
        """
        return self.val_dl
