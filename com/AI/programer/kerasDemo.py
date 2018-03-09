#!/usr/bin/python
#encoding=utf-8

import numpy as np
import pandas as pd
#神经网络可以映射成所有的非线性函数，首先对各节点对输入进行加权求和，
#通过一个非线性函数，实现模型对非线性处理如（sigmod,thanh,ReLu,softPlus）
#sigmod 经常会用到输出层，Relu深度网络经常会用，tanh，一定程度上可以代替sigmod,


#Anaconda,是python好用对工具，包含python常用对模块
"""
利用keras构建基本的神经网络（两层，求和，relu,求和sigmoid）进行分类
"""
from keras.models import Sequential #认为是神经网络的各个层容器
from keras.optimizers import SGD   #随机梯度下降法的优化方法
from keras.layers import  Dense,Activation #dense q求和的层
from sklearn.datasets import load_iris

iris = load_iris()

#运用分类器进行标签化
#print iris["target"]
from sklearn.preprocessing import LabelBinarizer
print LabelBinarizer().fit_transform(iris["target"])
from sklearn.cross_validation import train_test_split

#分为训练集和测试集
#random_state=1 是指随机状态，随机状态为固定值

train_data,test_data,train_target,test_target=train_test_split(iris.data,iris.target,test_size=0.2,random_state=1)

#值没赋上，进行标签化；因为输出标签做啦变动，所以要进行标签化
labels_train = LabelBinarizer().fit_transform(train_target)
labels_test = LabelBinarizer().fit_transform(test_target)
model=Sequential(
    [
        #第一层输出有5个，
        Dense(5,input_dim=4),
        #激活函数
        Activation("relu"),
        Dense(3),
        #激活函数
        Activation("sigmoid")
    ]

)
#构建模型
#还有另一种方式，定义结构
#model2 = Sequential()
#model2.add(Dense(5,input_dim=4))

#优化器随机梯度算法  lr :学习率，decay :学习衰减因子lr_i = lr_start * 1.0 / (1.0 + decay * i)
#decay越小，学习率衰减地越慢，当decay = 0时，学习率保持不变。 ，momentum ：动量因子，
#当使用冲量时，则把每次x的更新量v考虑为本次的梯度下降量- dx * lr与上次x的更新量v乘上一个介于[0, 1]的因子momentum的和
# ，即v = - dx * lr + v * momemtum。
#从公式上可看出： 。
sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
#分类交叉熵：分类交叉熵损失也被称为负对数似然，可用于测量两种概率分布（通常是真实标签和预测标签）之间的相似性。它可用 L =
#-sum(y * log(y_prediction)) 表示，其中 y 是真实标签的概率分布（通常是一个one-hotvector），
# y_prediction 是预测标签的概率分布，通常来自于一个 softmax。
model.compile(optimizer=sgd,loss="categorical_crossentropy")

#训练数据 nb_epoch 训练几轮，batch_size 训练一次多少数据
model.fit(train_data,labels_train,nb_epoch=200,batch_size=40)

print model.predict_classes(test_data)

#
#model.sample_weights("./data/w")
#model.load_weights("./data/w")

