import jieba
import pandas as pd


sent = []

f = f = open('train.txt','r',encoding='utf8')
# f = pd.read_csv('train.tsv',delimiter='\t',header=0).values
ff = f.readlines()
for i in ff:
    l = i.split('\t')
    sent.append(l[0])
    sent.append(l[1])

f = open('test.txt','r',encoding='utf8')
ff = f.readlines()
for i in ff:
    l = i.split('\t')
    sent.append(l[0])
    sent.append(l[1])

f = open('dev.txt','r',encoding='utf8')
ff = f.readlines()
for i in ff:
    l = i.split('\t')
    sent.append(l[0])
    sent.append(l[1])

word_list = []
for i in sent:
    sent_piece = jieba.lcut(i)
    for w in sent_piece:
        if w not in word_list:
            word_list.append(w)

print(len(word_list))
t = open('word_vocab.txt','a',encoding='utf8')
for i in word_list:
    t.write(i + '\n')