with open('train_hn.txt','r',encoding='utf8') as f:
    hn = f.readlines()
sent1 = []
sent2 = []
label = []
for i in hn:
    s = i.rstrip().split('\t')
    sent1.append(s[0])
    sent2.append(s[1])
    label.append(s[2])

f2 = open('train.txt','r',encoding='utf8')
ff = f2.readlines()
for i in ff:
    l = i.rstrip().split('\t')
    sent1.append(l[0])
    sent2.append(l[1])
    label.append(l[2])

with open('train_all.txt','a',encoding='utf8') as writer:
    for s1,s2,l in zip(sent1,sent2,label):
        writer.write(s1 + '\t' + s2 + '\t' + l + '\n')