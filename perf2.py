# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: performance.py
@Time: 2021/4/1 11:48
"""
import os
import json
import torch
import argparse
import numpy as np
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText
from tqdm import tqdm
from w3lib.html import remove_tags

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='txt/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='txt/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=2, type=int)
parser.add_argument('--bert-config-path', default='bert/cased_L-12_H-768_A-12/bert_config.json', type=str)
parser.add_argument('--bert-checkpoint-path', default='bert/cased_L-12_H-768_A-12/bert_model.ckpt', type=str)
parser.add_argument('--bert-dict-path', default='bert/cased_L-12_H-768_A-12/vocab.txt', type=str)

device = torch.device("cpu")

# ── 10 hardcoded test sentences ───────────────────────────────────────────────
DEMO_SENTENCES = [
    "the cat sat on the mat",
    "a quick brown fox jumps over the lazy dog",
    "she sells sea shells by the sea shore",
    "how much wood would a woodchuck chuck",
    "the weather outside is frightful but the fire is delightful",
    "to be or not to be that is the question",
    "all that glitters is not gold",
    "the early bird catches the worm",
    "a stitch in time saves nine",
    "actions speak louder than words",
]
# ─────────────────────────────────────────────────────────────────────────────


def sentences_to_tensor(sentences, token_to_idx, max_length, pad_idx, start_idx, end_idx):
    """Tokenise a list of whitespace-separated sentences into a padded LongTensor."""
    batch = []
    for sent in sentences:
        tokens = [token_to_idx.get(w, token_to_idx.get("<UNK>", pad_idx))
                  for w in sent.lower().split()]
        # Truncate, then add START / END markers to match training format
        tokens = [start_idx] + tokens[:max_length - 2] + [end_idx]
        # Pad to max_length
        tokens += [pad_idx] * (max_length - len(tokens))
        batch.append(tokens)
    return torch.LongTensor(batch)


def performance(args, SNR, net):
    bleu_score_1gram = BleuScore(0, 0, 0, 1)

    StoT = SeqtoText(token_to_idx, end_idx)

    # Build the fixed sentence tensor once
    sents_tensor = sentences_to_tensor(
        DEMO_SENTENCES, token_to_idx,
        args.MAX_LENGTH, pad_idx, start_idx, end_idx
    ).to(device)                          # shape: (10, MAX_LENGTH)

    score = []
    net.eval()
    with torch.no_grad():
        for epoch in range(args.epochs):
            Tx_word = []
            Rx_word = []

            for snr in tqdm(SNR):
                noise_std = SNR_to_noise(snr)

                out = greedy_decode(net, sents_tensor, noise_std, args.MAX_LENGTH,
                                    pad_idx, start_idx, args.channel)

                # Reconstructed sentences
                sentences = out.cpu().numpy().tolist()
                rx = list(map(StoT.sequence_to_text, sentences))

                # Original sentences (re-decoded from the token tensor for consistency)
                target_sent = sents_tensor.cpu().numpy().tolist()
                tx = list(map(StoT.sequence_to_text, target_sent))

                # ── Print original vs reconstructed for the first SNR in each epoch ──
                if snr == SNR[0]:
                    print(f"\n[Epoch {epoch+1}  SNR={snr} dB]")
                    print(f"{'ORIGINAL':<55}  RECONSTRUCTED")
                    print("-" * 110)
                    for orig, recon in zip(tx, rx):
                        print(f"{orig:<55}  {recon}")

                Tx_word.append(rx)
                Rx_word.append(tx)

            bleu_score = []
            for sent1, sent2 in zip(Tx_word, Rx_word):
                bleu_score.append(bleu_score_1gram.compute_blue_score(sent1, sent2))
            bleu_score = np.array(bleu_score)
            bleu_score = np.mean(bleu_score, axis=1)
            score.append(bleu_score)

    score1 = np.mean(np.array(score), axis=0)
    return score1


if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [18]

    args.vocab_file = 'data/' + args.vocab_file
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    """ define model """
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                    num_vocab, num_vocab, args.d_model, args.num_heads,
                    args.dff, 0.1).to(device)

    model_paths = []
    for fn in os.listdir(args.checkpoint_path):
        if not fn.endswith('.pth'):
            continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1])
        model_paths.append((os.path.join(args.checkpoint_path, fn), idx))

    model_paths.sort(key=lambda x: x[1])
    model_path, _ = model_paths[-1]
    checkpoint = torch.load(model_path , map_location=torch.device('cpu'))
    deepsc.load_state_dict(checkpoint)
    print('model load!')

    bleu_score = performance(args, SNR, deepsc)
    print("\nBLEU scores per SNR:", bleu_score)