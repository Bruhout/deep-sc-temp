# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
performance.py

--- DRY-RUN / LOCAL-DEV MODE ---
The performance() function is replaced with a stub that returns plausible
dummy BLEU scores without running greedy_decode or loading real data.
The argument parser, model construction, and checkpoint-loading logic are
unchanged so the full __main__ entry point is exercised.
"""

import os
import json
import torch
import argparse
import numpy as np
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText
from tqdm import tqdm
from sklearn.preprocessing import normalize
from w3lib.html import remove_tags

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir',             default='europarl/train_data.pkl',                    type=str)
parser.add_argument('--vocab-file',            default='europarl/vocab.json',                         type=str)
parser.add_argument('--checkpoint-path',       default='checkpoints/deepsc-Rayleigh',                type=str)
parser.add_argument('--channel',               default='Rayleigh',                                    type=str)
parser.add_argument('--MAX-LENGTH',            default=30,   type=int)
parser.add_argument('--MIN-LENGTH',            default=4,    type=int)
parser.add_argument('--d-model',               default=128,  type=int)
parser.add_argument('--dff',                   default=512,  type=int)
parser.add_argument('--num-layers',            default=4,    type=int)
parser.add_argument('--num-heads',             default=8,    type=int)
parser.add_argument('--batch-size',            default=64,   type=int)
parser.add_argument('--epochs',                default=2,    type=int)
parser.add_argument('--bert-config-path',
    default='bert/cased_L-12_H-768_A-12/bert_config.json',   type=str)
parser.add_argument('--bert-checkpoint-path',
    default='bert/cased_L-12_H-768_A-12/bert_model.ckpt',    type=str)
parser.add_argument('--bert-dict-path',
    default='bert/cased_L-12_H-768_A-12/vocab.txt',          type=str)
# Dry-run flag
parser.add_argument('--dry-run', action='store_true', default=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_SNR_DEFAULT = [0, 3, 6, 9, 12, 15, 18]


def performance(args, SNR, net):
    """
    [DRY-RUN] Returns synthetic BLEU scores of the correct shape
    (len(SNR),) without running any real inference.

    The returned array mirrors the shape produced by the real implementation
    so all downstream code (printing, plotting, logging) works unchanged.
    """
    print("[DRY-RUN] performance() – returning synthetic BLEU scores.")
    # Simulate a plausible SNR→BLEU curve: score improves with higher SNR
    snr_arr   = np.array(SNR, dtype=float)
    # Sigmoid-shaped curve between ~0.3 and ~0.9, plus small noise
    rng       = np.random.default_rng(7)
    base      = 0.3 + 0.6 / (1.0 + np.exp(-0.15 * (snr_arr - 9)))
    mock_bleu = base + rng.normal(0, 0.01, size=base.shape)
    mock_bleu = np.clip(mock_bleu, 0.0, 1.0)
    return mock_bleu


if __name__ == '__main__':
    args = parser.parse_args()
    SNR  = _SNR_DEFAULT

    # ------------------------------------------------------------------
    # Vocab setup (mirrors main.py; falls back to synthetic vocab)
    # ------------------------------------------------------------------
    vocab_path = args.vocab_file
    prefixed   = '/import/antennas/Datasets/hx301/' + args.vocab_file
    if os.path.exists(prefixed):
        vocab_path = prefixed

    if os.path.exists(vocab_path):
        vocab        = json.load(open(vocab_path, 'rb'))
        token_to_idx = vocab['token_to_idx']
    else:
        print("[DRY-RUN] Vocab file not found – generating synthetic vocab.")
        from dataset import _MOCK_VOCAB_SIZE
        token_to_idx = {
            '<PAD>':   0,
            '<START>': 1,
            '<END>':   2,
            '<UNK>':   3,
            **{f'word_{i}': i + 4 for i in range(_MOCK_VOCAB_SIZE - 4)}
        }

    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab    = len(token_to_idx)
    pad_idx      = token_to_idx["<PAD>"]
    start_idx    = token_to_idx["<START>"]
    end_idx      = token_to_idx["<END>"]

    # ------------------------------------------------------------------
    # Model construction – identical to production
    # ------------------------------------------------------------------
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                    num_vocab, num_vocab, args.d_model, args.num_heads,
                    args.dff, 0.1).to(device)

    # ------------------------------------------------------------------
    # Checkpoint loading – skipped gracefully when no checkpoints exist
    # ------------------------------------------------------------------
    if os.path.exists(args.checkpoint_path):
        model_paths = []
        for fn in os.listdir(args.checkpoint_path):
            if not fn.endswith('.pth'):
                continue
            idx = int(os.path.splitext(fn)[0].split('_')[-1])
            model_paths.append((os.path.join(args.checkpoint_path, fn), idx))

        if model_paths:
            model_paths.sort(key=lambda x: x[1])
            model_path, _ = model_paths[-1]
            checkpoint    = torch.load(model_path, map_location=device)
            deepsc.load_state_dict(checkpoint)
            print(f'[DRY-RUN] Model loaded from {model_path}')
        else:
            print('[DRY-RUN] No checkpoints found – using random weights.')
    else:
        print('[DRY-RUN] Checkpoint directory not found – using random weights.')

    bleu_score = performance(args, SNR, deepsc)
    print("BLEU scores per SNR:")
    for snr, score in zip(SNR, bleu_score):
        print(f"  SNR={snr:3d} dB  →  BLEU-1={score:.4f}")
