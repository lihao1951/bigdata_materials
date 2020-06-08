# 相似度综述

## 文本表示

### 局部敏感Hash

1. minhash
2. simhash

### 文本向量空间表示

1. 词袋模型

   词袋模型为最简单的文本表示模型

2. TFIDF

   在词袋模型的基础上利用词的全文概率信息（词频+逆文档频率）来代替词的统计信息

### 主题向量

1. LSA
2. PLSA
3. LDA

### 词汇向量

1. Word2vec

   CBOW和Skip-Gram两种模型

2. ELMO

3. Glove

   与word2vec相比较而言，加入了全局的位置信息embedding

4. GPT

5. BERT

   利用Transformer来生成很深的网络，对输入的文本进行编码

## 计算方法

1. 余弦距离
2. WMD距离
3. 欧氏距离

## 依赖的关键词抽取

1. TFIDF抽取关键词
2. TextRank抽取关键词
3. LDA关键词分布提取