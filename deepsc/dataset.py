# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: EurDataset.py
@Time: 2021/3/31 23:20

--- DRY-RUN / LOCAL-DEV MODE ---
EurDataset now generates synthetic token sequences instead of reading from
disk.  The collate_data function is unchanged so the DataLoader pipeline
behaves identically to production.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Synthetic-data parameters – keep in sync with vocab size used in main.py
# ---------------------------------------------------------------------------
_MOCK_VOCAB_SIZE  = 1000   # must be <= num_vocab used when building DeepSC
_MOCK_NUM_SAMPLES = 256    # total sentences returned by the dataset
_MOCK_MIN_LEN     = 5
_MOCK_MAX_LEN     = 20
_MOCK_SEED        = 42

_rng = np.random.default_rng(_MOCK_SEED)


class EurDataset(Dataset):
    """
    DROP-IN REPLACEMENT for the real Europarl dataset.

    Returns randomly generated integer sequences whose lengths fall within
    [_MOCK_MIN_LEN, _MOCK_MAX_LEN].  Token indices are drawn from
    [4, _MOCK_VOCAB_SIZE) so that the four special tokens (0-3) are never
    produced as body tokens.
    """

    def __init__(self, split='train'):
        # Slightly more samples for train to mirror real proportions
        n = _MOCK_NUM_SAMPLES if split == 'test' else _MOCK_NUM_SAMPLES * 4
        lengths = _rng.integers(_MOCK_MIN_LEN, _MOCK_MAX_LEN + 1, size=n)
        self.data = [
            _rng.integers(4, _MOCK_VOCAB_SIZE, size=l).tolist()
            for l in lengths
        ]
        print(f"[DRY-RUN] EurDataset(split='{split}') – "
              f"{len(self.data)} synthetic sentences loaded.")

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def collate_data(batch):
    """Unchanged from production – pads a batch to the longest sequence."""
    batch_size = len(batch)
    max_len    = max(len(x) for x in batch)
    sents      = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x), reverse=True)

    for i, sent in enumerate(sort_by_len):
        sents[i, :len(sent)] = sent

    return torch.from_numpy(sents)
