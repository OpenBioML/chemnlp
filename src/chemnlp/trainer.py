"""A custom trainer for modifying data sampling behaviour"""
from typing import Optional
import torch
import datasets
from torch.utils.data import DataLoader, sampler
from transformers import Trainer
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available

class LLcheMTrainer(Trainer):
    def __init__(
        self,
        sampler: Optional[sampler.Sampler] = None,
        **kwargs
    ):
        """
        Rewritten over from transformers 4.30.2
        * custom sampler
        * all other kwargs get passed as normal
        """
        super().__init__(**kwargs)
        self.sampler = sampler

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Uses default transformers behaviour unless a custom sampler is provided.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self._train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )
    