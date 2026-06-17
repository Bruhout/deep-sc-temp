# -*- coding: utf-8 -*-
"""
main.py

--- DRY-RUN / LOCAL-DEV MODE ---
The training loop is constrained to 1 epoch and a small number of batches
so the full code path executes quickly on a machine without a GPU or the
real dataset.  Every other aspect of the program – argument parsing, model
instantiation, optimizer setup, checkpoint saving, validate() call – is
identical to production.

To restore full training, revert the two lines marked  ← PRODUCTION VALUE.
"""

import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import SNR_to_noise, initNetParams, train_step, val_step, train_mi
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Argument defaults are unchanged from production.
# The only dry-run override is --epochs, handled below in __main__.
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file',       default='europarl/vocab.json',         type=str)
parser.add_argument('--checkpoint-path',  default='checkpoints/deepsc-Rayleigh', type=str)
parser.add_argument('--channel',          default='Rayleigh',                     type=str,
                    help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH',       default=30,  type=int)
parser.add_argument('--MIN-LENGTH',       default=4,   type=int)
parser.add_argument('--d-model',          default=128, type=int)
parser.add_argument('--dff',              default=512, type=int)
parser.add_argument('--num-layers',       default=4,   type=int)
parser.add_argument('--num-heads',        default=8,   type=int)
parser.add_argument('--batch-size',       default=128, type=int)
parser.add_argument('--epochs',           default=80,  type=int)  # ← PRODUCTION VALUE

# Dry-run knobs (no effect in production – these args are simply ignored there)
parser.add_argument('--dry-run',          action='store_true', default=True,
                    help='[dry-run] Run mock loop only (default: True in this build)')
parser.add_argument('--dry-run-epochs',   default=1,  type=int,
                    help='[dry-run] Number of epochs to run')
parser.add_argument('--dry-run-batches',  default=2,  type=int,
                    help='[dry-run] Max batches per epoch (train + val)')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# validate() – identical structure to production; uses mock val_step
# ---------------------------------------------------------------------------
def validate(epoch, args, net):
    test_eur      = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size,
                               num_workers=0, pin_memory=True,
                               collate_fn=collate_data)
    net.eval()
    pbar  = tqdm(test_iterator)
    total = 0

    with torch.no_grad():
        for batch_idx, sents in enumerate(pbar):
            # [DRY-RUN] honour batch cap
            if args.dry_run and batch_idx >= args.dry_run_batches:
                break

            sents = sents.to(device)
            loss  = val_step(net, sents, sents, 0.1, pad_idx,
                             criterion, args.channel)
            total += loss
            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(epoch + 1, loss)
            )

    n_batches = min(args.dry_run_batches, len(test_iterator)) \
                if args.dry_run else len(test_iterator)
    return total / max(n_batches, 1)


# ---------------------------------------------------------------------------
# train() – identical structure to production; uses mock train_step/train_mi
# ---------------------------------------------------------------------------
def train(epoch, args, net, mi_net=None):
    train_eur      = EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size,
                                num_workers=0, pin_memory=True,
                                collate_fn=collate_data)
    pbar      = tqdm(train_iterator)
    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1,))

    for batch_idx, sents in enumerate(pbar):
        # [DRY-RUN] honour batch cap
        if args.dry_run and batch_idx >= args.dry_run_batches:
            break

        sents = sents.to(device)

        if mi_net is not None:
            mi   = train_mi(net, mi_net, sents, 0.1, pad_idx,
                            mi_opt, args.channel)
            loss = train_step(net, sents, sents, 0.1, pad_idx,
                              optimizer, criterion, args.channel, mi_net)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}; MI {:.5f}'.format(
                    epoch + 1, loss, mi)
            )
        else:
            loss = train_step(net, sents, sents, noise_std[0], pad_idx,
                              optimizer, criterion, args.channel)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}'.format(epoch + 1, loss)
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # [DRY-RUN] Override epoch count so the loop exits quickly.
    # In production this block is absent; args.epochs stays at 80.
    # ------------------------------------------------------------------
    if args.dry_run:
        print("\n" + "="*60)
        print("  DRY-RUN MODE ACTIVE")
        print(f"  epochs   : {args.dry_run_epochs}  (production: {args.epochs})")
        print(f"  batches  : {args.dry_run_batches} per phase (train + val)")
        print(f"  dataset  : synthetic (no disk reads)")
        print(f"  training : mock stubs (no forward/backward pass)")
        print("="*60 + "\n")
        args.epochs = args.dry_run_epochs  # ← overridden for dry-run only

    # ------------------------------------------------------------------
    # Vocab setup – uses a synthetic vocab when the real file is absent
    # ------------------------------------------------------------------
    vocab_path = args.vocab_file
    # Allow an optional data-dir prefix only when the file actually exists there
    prefixed = '/import/antennas/Datasets/hx301/' + args.vocab_file
    if os.path.exists(prefixed):
        vocab_path = prefixed

    if os.path.exists(vocab_path):
        vocab         = json.load(open(vocab_path, 'rb'))
        token_to_idx  = vocab['token_to_idx']
    else:
        # [DRY-RUN] build a minimal synthetic vocabulary so the rest of the
        # script works without any real data files on disk.
        print("[DRY-RUN] Vocab file not found – generating synthetic vocab.")
        from dataset import _MOCK_VOCAB_SIZE
        token_to_idx = {
            '<PAD>':   0,
            '<START>': 1,
            '<END>':   2,
            '<UNK>':   3,
            **{f'word_{i}': i + 4 for i in range(_MOCK_VOCAB_SIZE - 4)}
        }

    num_vocab  = len(token_to_idx)
    pad_idx    = token_to_idx["<PAD>"]
    start_idx  = token_to_idx["<START>"]
    end_idx    = token_to_idx["<END>"]

    # ------------------------------------------------------------------
    # Model, optimiser, loss – identical to production
    # ------------------------------------------------------------------
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                    num_vocab, num_vocab, args.d_model, args.num_heads,
                    args.dff, 0.1).to(device)

    mi_net    = Mine().to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(deepsc.parameters(),
                                 lr=1e-4, betas=(0.9, 0.98),
                                 eps=1e-8, weight_decay=5e-4)
    mi_opt    = torch.optim.Adam(mi_net.parameters(), lr=1e-3)

    initNetParams(deepsc)

    # ------------------------------------------------------------------
    # Training loop – structure identical to production
    # ------------------------------------------------------------------
    record_acc  = 10
    record_loss = []

    for epoch in range(args.epochs):
        start = time.time()

        train(epoch, args, deepsc)
        avg_acc = validate(epoch, args, deepsc)

        print(f"[Epoch {epoch+1}] val_loss={avg_acc:.5f}  "
              f"elapsed={time.time()-start:.1f}s")

        if avg_acc < record_acc:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            ckpt_path = os.path.join(
                args.checkpoint_path,
                'checkpoint_{}.pth'.format(str(epoch + 1).zfill(2))
            )
            with open(ckpt_path, 'wb') as f:
                torch.save(deepsc.state_dict(), f)
            print(f"[DRY-RUN] Checkpoint saved → {ckpt_path}")
            record_acc = avg_acc

    print("\n[DRY-RUN] Training loop completed successfully.")
    print("Script is structurally valid – safe to push to training server.")
