# ELMo

核心思想：

- 基于语言模型的训练：使用LSTM作为基石
- 启发来自于深度学习中的层次表示（Hierachical Representation）

两者结合为Deep BI-LSTM

解决的问题是：对于一个单词，动态的学出在上下文种的词向量

ELMo不是完全双向LSTM

为了搭建完整的双向LSTM语言模型，后来出来了XLNet模型

