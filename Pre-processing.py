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
    path ='data/chinese/AFQMC/word_vocab.txt'
    vocab = [line.strip() for line in open(path, encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word

def word2index(sentence1,sentence2):
    word2idx,idx2word = load_word_vocab()
    s1_index_all,s2_index_all = [],[]
    s1_mask_all,s2_mask_all = [],[]
    mat = []
    for s1,s2 in tqdm(zip(sentence1,sentence2)):
        res =[[0] * max_len for _ in range(max_len)]
        s1_ = jieba.lcut(s1)
        s2_ = jieba.lcut(s2)
        s1_index, s2_index = [], []
        for s1_word in s1_:
            if len(s1_word)>0 and s1_word in word2idx.keys():
                s1_index.append(word2idx[s1_word])
            else:
                s1_index.append(1)

        for s2_word in s2_:
            if len(s2_word)>0 and s2_word in word2idx.keys():
                s2_index.append(word2idx[s2_word])
            else:
                s2_index.append(1)
        if len(s1_index) >= args.max_len:
            s1_index = s1_index[:args.max_len]
            s1_mask = [True] * args.max_len
        else:
            s1_mask = [True] * len(s1_index) + [False] * (max_len - len(s1_index))
            s1_index = s1_index + [0] * (max_len-len(s1_index))


        if len(s2_index) >= args.max_len:
            s2_index = s2_index[:max_len]
            s2_mask = [True] * args.max_len
        else:
            s2_mask = [True] * len(s2_index) + [False] * (max_len - len(s2_index))
            s2_index = s2_index + [0] * (max_len-len(s2_index))


        for i in range(max_len):
            for j in range(max_len):
                if is_sememe(idx2word[s1_index[i]], idx2word[s2_index[j]]) and s1_index != 0 and s2_index != 0 and s1_index != 0 and s2_index != 0:
                    res[i][j] = 1
        s1_index_all.append(s1_index)
        s1_mask_all.append(s1_mask)
        s2_index_all.append(s2_index)
        s2_mask_all.append(s2_mask)
        mat.append(res)
    return s1_index_all,s1_mask_all,s2_index_all,s2_mask_all,mat


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




# class LoadTestData(Dataset):
#     def __init__(self,dir,max_len=args.max_len):
#         s1,s2,label = get_txt(dir)
#         self.sent1 = s1[:]
#         self.sent2 = s2[:]
#         self.tag = label[:]
#         self.max_len = max_len
#         self.s1_index,self.s2_index =  word2index(self.sent1,self.sent2)
#
#
#     def __getitem__(self, i):
#         return self.s1_index[i], self.s2_index[i], self.tag[i]
#     def __len__(self):
#         return len(self.tag)
#
# def collate_t(batch):
#     s1 = torch.LongTensor([item[0] for item in batch])
#     s2 = torch.LongTensor([item[1] for item in batch])
#     label = torch.LongTensor([item[2] for item in batch])
#
#     return s1,s2,label
if __name__ == '__main__':
    import pickle
    # sent1, sent2, label = get_txt('data/chinese/AFQMC/train.txt')
    # s1_index, s1_mask, s2_index, s2_mask, mat = word2index(sent1, sent2)
    # data = [s1_index, s1_mask, s2_index, s2_mask, mat, label]
    # f = open('AFQMC_train.pickle','wb')
    # pickle.dump(data,f)

    sent1, sent2, label = get_txt('data/chinese/BQ/train.txt')
    s1_index, s1_mask, s2_index, s2_mask, mat = word2index(sent1, sent2)
    data = [s1_index, s1_mask, s2_index, s2_mask, mat, label]
    f = open('BQ_train.pickle','wb')
    pickle.dump(data,f)