import os
import json
import torch
import numpy as np
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText
from tqdm import tqdm
from sklearn.preprocessing import normalize
from w3lib.html import remove_tags

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------------------------------------------------------------
# Global configuration (previously command-line arguments via argparse)
# ----------------------------------------------------------------------------
DATA_DIR = 'txt/train_data.pkl'
VOCAB_FILE = 'data/txt/vocab.json'
CHECKPOINT_PATH = 'checkpoints/deepsc-Rayleigh'
CHANNEL = 'Rayleigh'
MAX_LENGTH = 30
MIN_LENGTH = 4
D_MODEL = 128
DFF = 512
NUM_LAYERS = 4
NUM_HEADS = 8
BATCH_SIZE = 64
EPOCHS = 2
BERT_CONFIG_PATH = 'bert/cased_L-12_H-768_A-12/bert_config.json'
BERT_CHECKPOINT_PATH = 'bert/cased_L-12_H-768_A-12/bert_model.ckpt'
BERT_DICT_PATH = 'bert/cased_L-12_H-768_A-12/vocab.txt'

SNR = [0, 3, 6, 9, 12, 15, 18]

device = torch.device("cpu")

# Load tokenizer and model fine-tuned on MNLI
model_name = "roberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()


def performance(snr_list, net):
    bleu_score_1gram = BleuScore(1, 0, 0, 0)

    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=BATCH_SIZE, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)

    StoT = SeqtoText(token_to_idx, end_idx)
    score = []
    nli_score = []
    score2 = []
    net.eval()
    with torch.no_grad():
        for epoch in range(EPOCHS):
            received = []
            transmitted = []

            for snr in tqdm(snr_list):
                word = []
                target_word = []
                noise_std = SNR_to_noise(snr)

                # iterator returns a batch with mutiple sentences
                for batch_idx, batch in enumerate(test_iterator):
                    if batch_idx % 100000 != 0:
                        continue

                    batch = batch.to(device)
                    target = batch

                    out = greedy_decode(net, batch, noise_std, MAX_LENGTH, pad_idx,
                                        start_idx, CHANNEL)

                    sentences = out.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, sentences))
                    word = word + result_string

                    target_sent = target.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, target_sent))
                    target_word = target_word + result_string

                received.append(word)
                transmitted.append(target_word)

                # print(f"_______________current snr is: {snr}_______________")
                # for i in range(len(word)):
                    # print("Transmitted: " + target_word[i])
                    # print("Received: " + word[i])

                    # inputs = tokenizer(target_word[i], word[i], return_tensors="pt", truncation=True)

                    # with torch.no_grad():
                    #     logits = model(**inputs).logits

                    # probs = torch.softmax(logits, dim=-1)
                    # probs = np.array(probs)
                    # probs = np.mean(probs, axis=1)

                    # labels = ["contradiction", "neutral", "entailment"]
                    # pred_idx = probs.argmax(dim=-1).item()

                    # print(f"Prediction: {labels[pred_idx]}")
                    # print({labels[i]: round(probs[0][i].item(), 4) for i in range(len(labels))})

            bleu_score = []
            for sent1, sent2 in zip(received, transmitted):
                bleu_score.append(bleu_score_1gram.compute_blue_score(sent1, sent2)) # 7*num_sent
            bleu_score = np.array(bleu_score)
            bleu_score = np.mean(bleu_score, axis=1)
            score.append(bleu_score)

    score1 = np.mean(np.array(score), axis=0)

    return score1


if __name__ == '__main__':
    vocab = json.load(open(VOCAB_FILE, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    """ define optimizer and loss function """
    deepsc = DeepSC(NUM_LAYERS, num_vocab, num_vocab,
                        num_vocab, num_vocab, D_MODEL, NUM_HEADS,
                        DFF, 0.1).to(device)

    checkpoint = torch.load("checkpoints/deepsc-Rayleigh/checkpoint_80_europarl.pth", map_location=device)
    deepsc.load_state_dict(checkpoint)
    print('model load!')

    bleu_score = performance(SNR, deepsc)
    print("Bleu scores for this epoch are: ")
    print(bleu_score)

    #similarity.compute_similarity(sent1, real)