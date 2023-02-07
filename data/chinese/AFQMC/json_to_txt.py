import json

ff = open('test.json','r',encoding='utf8')
data = ff.readlines()
ans = []
for i in data:
    d = json.loads(i)
    sent1 = d['sentence1']
    sent2 = d['sentence2']
    # label = d['label']
    res = sent1 + '\t' + sent2 + '\t'
    ans.append(res)
for i in ans:
    with open('test.txt','a',encoding='utf8') as aa:
        aa.write(i + '\n')
