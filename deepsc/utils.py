# -*- coding: utf-8 -*-
"""
utils.py

--- DRY-RUN / LOCAL-DEV MODE ---
train_step, val_step, train_mi, and greedy_decode are replaced with stubs
that return plausible dummy values without performing any real forward or
backward passes.  Every other utility (BleuScore, LabelSmoothing, NoamOpt,
SeqtoText, Channels, masking helpers, SNR_to_noise, PowerNormalize,
initNetParams, loss_function) is IDENTICAL to production so that imports
and any code paths that call them continue to work correctly.
"""

import os
import math
import time
import torch
import torch.nn as nn
import numpy as np
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu
from models.mutual_info import sample_batch, mutual_information

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Unchanged production utilities
# ---------------------------------------------------------------------------

class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4

    def compute_blue_score(self, real, predicted):
        score = []
        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1).split()
            sent2 = remove_tags(sent2).split()
            score.append(sentence_bleu([sent1], sent2,
                          weights=(self.w1, self.w2, self.w3, self.w4)))
        return score


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion   = nn.CrossEntropyLoss()
        self.padding_idx = padding_idx
        self.confidence  = 1.0 - smoothing
        self.smoothing   = smoothing
        self.size        = size
        self.true_dist   = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer    = optimizer
        self._step        = 0
        self.warmup       = warmup
        self.factor       = factor
        self.model_size   = model_size
        self._rate        = 0
        self._weight_decay = 0

    def step(self):
        self._step += 1
        rate         = self.rate()
        weight_decay = self.weight_decay()
        for p in self.optimizer.param_groups:
            p['lr']           = rate
            p['weight_decay'] = weight_decay
        self._rate         = rate
        self._weight_decay = weight_decay
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        lr = self.factor * (
            self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5))
        )
        return lr

    def weight_decay(self, step=None):
        if step is None:
            step = self._step
        if step <= 3000:
            weight_decay = 1e-3
        elif step <= 9000:
            weight_decay = 0.0005
        else:
            weight_decay = 1e-4
        weight_decay = 0
        return weight_decay


class SeqtoText:
    def __init__(self, vocb_dictionary, end_idx):
        self.reverse_word_map = dict(zip(vocb_dictionary.values(),
                                         vocb_dictionary.keys()))
        self.end_idx = end_idx

    def sequence_to_text(self, list_of_indices):
        words = []
        for idx in list_of_indices:
            if idx == self.end_idx:
                break
            words.append(self.reverse_word_map.get(idx))
        return ' '.join(words)


class Channels():

    def AWGN(self, Tx_sig, n_var):
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_var):
        shape  = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H      = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)
        return Rx_sig

    def Rician(self, Tx_sig, n_var, K=1):
        shape  = Tx_sig.shape
        mean   = math.sqrt(K / (K + 1))
        std    = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H      = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)
        return Rx_sig


def initNetParams(model):
    """Init net parameters."""
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape      = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)


def create_masks(src, trg, padding_idx):
    src_mask        = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor)
    trg_mask        = (trg == padding_idx).unsqueeze(-2).type(torch.FloatTensor)
    look_ahead_mask = subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
    combined_mask   = torch.max(trg_mask, look_ahead_mask)
    return src_mask.to(device), combined_mask.to(device)


def loss_function(x, trg, padding_idx, criterion):
    loss = criterion(x, trg)
    mask = (trg != padding_idx).type_as(loss.data)
    loss *= mask
    return loss.mean()


def PowerNormalize(x):
    x_square = torch.mul(x, x)
    power    = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)
    return x


def SNR_to_noise(snr):
    snr       = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std


# ---------------------------------------------------------------------------
# MOCK training / evaluation functions
# ---------------------------------------------------------------------------
# These stubs intentionally skip all forward/backward passes.
# They return realistic-looking scalar values so that the surrounding
# training loop logic (loss comparisons, checkpoint saving, tqdm display)
# runs exactly as it would in production.
#
# The model's nn.Module graph is built and initNetParams() IS called in
# main.py, so the architecture is fully instantiated and validated – only
# the actual data flow through it is skipped.
# ---------------------------------------------------------------------------

_mock_rng = np.random.default_rng(0)


def _mock_loss(base=5.0, noise=0.05):
    """Return a gently decreasing noisy loss so tqdm output looks natural."""
    return float(base + _mock_rng.normal(0, noise))


def train_step(model, src, trg, n_var, pad, opt, criterion, channel,
               mi_net=None):
    """
    [DRY-RUN] Skips forward/backward pass.
    Returns a plausible dummy training loss.
    """
    return _mock_loss(base=4.8)


def train_mi(model, mi_net, src, n_var, padding_idx, opt, channel):
    """
    [DRY-RUN] Skips mutual-information network update.
    Returns a plausible dummy MI lower-bound value.
    """
    return _mock_loss(base=0.5, noise=0.02)


def val_step(model, src, trg, n_var, pad, criterion, channel):
    """
    [DRY-RUN] Skips forward pass.
    Returns a plausible dummy validation loss.
    """
    return _mock_loss(base=4.9)


def greedy_decode(model, src, n_var, max_len, padding_idx, start_symbol,
                  channel):
    """
    [DRY-RUN] Skips encoder/decoder forward passes.
    Returns a tensor of the correct shape (batch_size, max_len) filled with
    the start symbol so that downstream SeqtoText conversion doesn't crash.
    """
    batch_size = src.size(0)
    # Fill with start_symbol; SeqtoText will produce empty-ish strings,
    # which is fine – BLEU scores will be ~0, as expected for a mock run.
    outputs = torch.ones(batch_size, max_len,
                         dtype=src.dtype, device=src.device) * start_symbol
    return outputs
