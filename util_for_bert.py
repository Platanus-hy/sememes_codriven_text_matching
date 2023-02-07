import jieba
import torch
import pandas as pd
from torch.utils.data import DataLoader,Dataset
from gensim.models import word2vec
import json
import re
from how_net import is_sememe
import args
from tqdm import tqdm
from args import *

def load_word_vocab():
    path ='data/chinese/bq_corpus/word_vocab.txt'
    vocab = [line.strip() for line in open(path, encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word

def word2index(sentence1,sentence2):
    word2idx,idx2word = load_word_vocab()
    s1_ = []
    s2_ = []
    mat = []
    for s1,s2 in tqdm(zip(sentence1,sentence2)):
        seg1 = jieba.lcut(s1)
        seg2 = jieba.lcut(s2)
        s1_.append(seg1)
        s2_.append(seg2)
        res =[[0] * max_len for _ in range(max_len)]

        for i in range(max_len):
            for j in range(max_len):
                if i<len(seg1) and j<len(seg2) and is_sememe(seg1[i], seg2[j]) :
                    res[i][j] = 1
        mat.append(res)
    return s1_,s2_,mat


def get_txt(dir):
    cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")
    ff = open(dir, 'r', encoding='utf8')
    sent1, sent2, label = [], [], []
    for line in ff.readlines():
        # i = json.load(line)
        i = line.rstrip().split('\t')
        sent1.append(cop.sub('', i[0]))
        sent2.append(cop.sub('', i[1]))
        label.append(int(cop.sub('',i[2])))

    return sent1,sent2,label


class LoadData(Dataset):
    def __init__(self,dir,max_len=args.max_len):
        s1,s2,label = get_txt(dir)
        self.sent1 = s1[:10]
        self.sent2 = s2[:10]
        self.label = label[:10]
        self.max_len = max_len
        self.s1,self.s2,self.mat =  word2index(self.sent1,self.sent2)

    def __getitem__(self, i):
        return self.s1[i],self.s2[i] ,self.label[i],self.mat[i]
    def __len__(self):
        return len(self.label)

def collate(batch):
    s1 = [item[0] for item in batch]
    s2 = [item[1] for item in batch]
    label = torch.LongTensor([item[2] for item in batch])
    mat = torch.LongTensor([item[3] for item in batch])
    return s1,s2,label,mat

# if __name__ == '__main__':
#     import pickle
#     # sent1, sent2, label = get_txt('data/chinese/AFQMC/train.txt')
#     # s1_index, s1_mask, s2_index, s2_mask, mat = word2index(sent1, sent2)
#     # data = [s1_index, s1_mask, s2_index, s2_mask, mat, label]
#     # f = open('AFQMC_train.pickle','wb')
#     # pickle.dump(data,f)
#
#     sent1, sent2, label = get_txt('data/chinese/AFQMC/dev.txt')
#     s1_index, s1_mask, s2_index, s2_mask, mat = word2index(sent1, sent2)
#     data = [s1_index, s1_mask, s2_index, s2_mask, mat, label]
#     f = open('AFQMC_test.pickle','wb')
#     pickle.dump(data,f)