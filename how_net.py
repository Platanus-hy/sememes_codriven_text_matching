import OpenHowNet
import jieba
# OpenHowNet.download()
hownet = OpenHowNet.HowNetDict()
def is_sememe(word1,word2):
    result_list = hownet.get_sememes_by_word(word1)
    res = []
    for i in range(len(result_list)):
        for j in result_list[i]['sememes'][:]:
            if j.zh not in res:
                res.append(j.zh)
    res2 = []
    result_list = hownet.get_sememes_by_word(word2)
    for i in range(len(result_list)):

        for j in result_list[i]['sememes'][:]:
            if j.zh not in res2:
                res2.append(j.zh)
    jiaoji = list(set(res) & set(res2))
    if '功能词' in jiaoji:
        jiaoji.remove('功能词')
    return jiaoji

if __name__ == '__main__':
    a = hownet.get_sememes_by_word('中国')
    b = hownet.get_sememes_by_word('华夏')
    print(a)



    # sent1 = '为什么借款后一直没有给我回拨电话'
    # sent2 = '怎么申请借款后没有打电话过来呢！'
    #
    # words1 = jieba.lcut(sent1)
    # words2 = jieba.lcut(sent2)
    # res = [[0] * len(words1) for _ in range(len(words2)) ]
    # for i in range(len(words1)):
    #     for j in range(len(words2)):
    #         if is_sememe(words1[i],words2[j]):
    #             res[i][j] = 1
    #             print(words1[i],words2[j])
    # print(res)
