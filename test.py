# """
# 该文件用来测试和改写 Transformers 提供的 API
# """

# from transformers import EsmTokenizer, EsmModel, EsmConfig

# tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
# """
# 加载预训练模型后，模型会报错 newly initialized: ['esm.pooler.dense.weight', 'esm.pooler.dense.bias']，
# 这里其实并不影响模型的使用，因为这部分权重是将句子的第一个单词 <bos> 的表示进行简单线性变化，作为句子表示，
# 该结果保存在 outputs.pooler_output 中，只要不使用这个参数就行了。
# """
# model = EsmModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
# print(model.config)

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# print(inputs.keys())
# outputs = model(output_attentions=True, output_hidden_states=True, **inputs)
# print(outputs.last_hidden_state.shape)
# print(outputs.pooler_output.shape)
# print(len(outputs.hidden_states))
# print(len(outputs.attentions))
# print(outputs.hidden_states[0].shape)
# print(outputs.attentions[0].shape)

# """
# 注意 outputs 是一个字典类型的输出，默认输出
# last_hidden_state : [batch_size, seq_len, hidden_states] 模型最后一层的输出
# pooler_output : [batch_size, hidden_states] 模型最后一层输出的池化结果，这里使用 <bos> 最后一层表示的线性映射，映射层随机初始化所以需要训练后使用

# output_hidden_states = True 模型额外返回每一层的表示，包括最初的编码层表示，所以层数会+1，使用 tuple 封装
# hidden_stats : [batch_size, seq_len, hidden_states]

# output_attentions = True 模型额外返回每一层的注意力权重矩阵，使用 tuple 封装
# attentions : [batch_size, head_nums, seq_len, seq_len]
# """


# print(outputs.keys())

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree

data = np.array([[0, 2, 1, 1], 
[0, 2, 1, 0], 
[1, 2, 1, 1], 
[2, 1, 1, 1], 
[2, 0, 0, 1],
[2, 0, 0, 0],
[1, 0, 0, 0],
[0, 1, 1, 1],
[0, 0, 0, 1],
[2, 1, 0, 1],
[0, 1, 0, 0],
[1, 1, 1, 0],
[1, 2, 0, 1]
])

clf = DecisionTreeClassifier(criterion="entropy")
y = [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]
clf.fit(data, y)
fn = ['Age', 'Incoming', 'Student', 'Credit Rating']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(clf, feature_names=fn)
plt.show()