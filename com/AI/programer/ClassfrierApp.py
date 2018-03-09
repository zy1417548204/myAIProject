#!/usr/bin/python
#encoding=utf-8

"""
调用自己实现的感知器算法，生成模型，并检验准确性
"""
file = "////"
import pandas as pd
#传统的csv文件（逗号分割符）有表头，
# 这次这个没有需要设置下header
df = pd.read_csv(file,header=None)

#df.head(10) 显示数据的前十行
import matplotlib.pyplot as plt
import numpy as np
import Perception

#提取0～100行第四列数据 ,按行显示出来
y = df.loc[0:100,4].values
#print y
#转换当前类型
y = np.where(y == 'Iris-setosa',-1,1)
#提取0～100行的数据，按列显示出来
"""
加载数据并进行数据展示
"""
X = df.iloc[0:100,[0,2]].values
plt.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='setosa')
plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label='haha')
plt.xlabel('花瓣长度')
plt.ylabel('花径长度')
plt.legend(loc='upper left')
plt.show()
"""
 利用加载数据进行模型训练
"""
from matplotlib.colors import ListedColormap
def plot_decision_regions(X,y,classifier,resolution=0.02):
    marker = ('s','x','o','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #
    x1_min,x1_max = X[:,0].min() - 1,X[:,0].max()
    x2_min,x2_max = X[:,1].min() - 1,X[:,1].max()

    print (x1_min,x1_max)
    print (x2_min,x2_max)
    #通过arange构造向量扩展成二维矩阵   每个值之剪相差resolution
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                          np.arange(x2_min,x2_max,resolution))
    #np.arange().shape 返回一个arange个数。
    #将xx1,xx2求值并转质
    z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    #print xx1.ravel()
    #print xx2.ravel()
    #print z

    z = z.reshape(xx1.shape)
    #countourf： 根据给出的数据去绘制出分界线
    plt.countourf(xx1,xx2,z,alpha = 0.4,cmap=cmap)
    #x轴的坐标对应最大值及最小值
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    #for 循环对两组区域打上标签说明
    for idx,cl in enumerate(np.unique()):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],alpha=0.8,c=cmap(idx),marker=-marker[idx],label=cl)

ppn = Perception(eta = 0.1,n_iter = 10)
ppn.fit(X,y)
plot_decision_regions(X,y,ppn,resolution=0.02)
plt.xlabel('花径长度')
plt.ylabel('花瓣长度')
plt.legend(loc= 'upper left')
plt.show()