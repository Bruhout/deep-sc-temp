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
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from w3lib.html import remove_tags

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='txt/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='txt/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type = int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=2, type = int)
parser.add_argument('--bert-config-path', default='bert/cased_L-12_H-768_A-12/bert_config.json', type = str)
parser.add_argument('--bert-checkpoint-path', default='bert/cased_L-12_H-768_A-12/bert_model.ckpt', type = str)
parser.add_argument('--bert-dict-path', default='bert/cased_L-12_H-768_A-12/vocab.txt', type = str)

device = torch.device("cuda:0")


# using pre-trained model to compute the sentence similarity
class Similarity():
    def __init__(self):
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')

    def compute_similarity(self, real, predicted):
        # Strip any html tags like the original did
        real      = [remove_tags(s) for s in real]
        predicted = [remove_tags(s) for s in predicted]

        vec1 = self.model.encode(real,      convert_to_numpy=True)
        vec2 = self.model.encode(predicted, convert_to_numpy=True)

        # Diagonal gives paired cosine similarity: real[i] vs predicted[i]
        scores = cosine_similarity(vec1, vec2)
        return np.diag(scores).tolist()


def performance(args, SNR, net):
    similarity = Similarity()

    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)

    StoT = SeqtoText(token_to_idx, end_idx)
    score = []
    score2 = []
    net.eval()
    with torch.no_grad():
        for epoch in range(args.epochs):
            Tx_word = []
            Rx_word = []

            for snr in tqdm(SNR):
                word = []
                target_word = []
                noise_std = SNR_to_noise(snr)

                for batch_idx, sents in enumerate(test_iterator):
                    if (batch_idx % 10 != 0):
                        continue
                    
                    sents = sents.to(device)
                    # src = batch.src.transpose(0, 1)[:1]
                    target = sents

                    out = greedy_decode(net, sents, noise_std, args.MAX_LENGTH, pad_idx,
                                        start_idx, args.channel)

                    sentences = out.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, sentences))
                    word = word + result_string

                    target_sent = target.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, target_sent))
                    target_word = target_word + result_string

                Tx_word.append(word)
                Rx_word.append(target_word)

            sim_score = []
            for i , (sent1, sent2) in enumerate(zip(Tx_word, Rx_word)):
                print(i)
                sim_score.append(similarity.compute_similarity(sent1, sent2)) # 7*num_sent
            sim_score = np.array(sim_score)
            sim_score = np.mean(sim_score, axis=1)
            score2.append(sim_score)

    score2 = np.mean(np.array(score2), axis=0)

    return score2

if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [0,3,6,9,12,15,18]

    args.vocab_file = 'data/' + args.vocab_file
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    """ define optimizer and loss function """
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                        num_vocab, num_vocab, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)

    checkpoint = torch.load("/content/checkpoints/deepsc-Rayleigh/checkpoint_80_europarl.pth")
    deepsc.load_state_dict(checkpoint)
    print('model load!')

    bert_score = performance(args, SNR, deepsc)
    
    print("bert_score:")
    print(bert_score)