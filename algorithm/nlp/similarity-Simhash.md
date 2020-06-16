# Simhash

## 原理

simhash属于局部敏感哈希的代表，与minhash同属于LSH(Location Sentiment Hash)。

Simhash的策略是

1. 首先对文本进行分词/抽取关键词，得到一组词语列表，比如"中华人民共和国万岁"，分词后得到"中华","人民","共和国","万岁"，我们记为$W$
2. 分别对$W$中的每个词$W_i$进行hash，这里选用hash算法的是

