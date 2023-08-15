import itertools
import math
import random
from typing import Iterator, List, Optional

import torch.distributed as dist
from torch.utils.data import Dataset, sampler


class SlidingWindowSampler(sampler.Sampler):
    """
    Sampler that restricts data loading to a subset of the dataset
        * given the incoming sorted, tokenised data
        * assume we want to sample in this current order
        * and repeat each data indices N times

                                dataset
    | 0  1  2  3 | 4  5  6  7 | 8  9  10  11 | 12  13  14  15 |
    |            |            |              |                |
         gpu 1        gpu 2         gpu 3          gpu 4

        * move the sliding window at varying pace

    | 1  2  3  4 | 5  6  7  8 | 9  10  11  12 | 13  14  15  16 |
    |            |            |               |                |
         gpu 1        gpu 2         gpu 3          gpu 4
    """

    def __init__(
        self,
        dataset: Dataset,
        num_repeats: int,
        num_replicas: Optional[int] = None,  # world size
        rank: Optional[int] = None,  # local rank
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_repeats = num_repeats
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas * self.num_repeats

    def _generate_windowed_indices(self) -> List[int]:
        # sliding window concatenation of size self.num_repeats
        indices = list(range(self.num_samples))
        sliding_windows = [
            indices[i : i + self.num_repeats] for i in range(0, self.num_samples)
        ]
        return list(itertools.chain.from_iterable(sliding_windows))

    def _handle_uneven_lengths(self, indices: List[int]) -> List[int]:
        # add extra samples to make it evenly divisible
        padding_size = self.total_size - len(indices)
        if 0 < padding_size:
            indices += random.sample(indices, k=padding_size)
        return indices

    def __iter__(self) -> Iterator:
        indices = self._generate_windowed_indices()

        if not self.drop_last:
            indices = self._handle_uneven_lengths(indices)
        else:
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        sub_indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(sub_indices) == self.num_samples * self.num_repeats

        return iter(sub_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
