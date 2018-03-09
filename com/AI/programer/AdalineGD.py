#!/usr/bin/python
#encoding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
class AdalineGD(object):
    """
    eta:float  学习率 介于0和1之间
    n_iter : int  对训练数据进行学习改进次数
    w_ : 一维向量
    error_ :存储每次迭代改进时，网络对数据进行错误判断对次数
    """

    def __index__(self,eta=0.01,n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
    def fit(self,X,y):
        #X: 二维数组 [n_samples,n_features]
        #n_samples:表示X中含有训练数据条目数
        #n_features :含有4个数据的一维向量，用于表示一条训练条目
        # y :一维向量
        #用于存储每一条条目对应的正确分类

        #权重全部初始化为零
        self.w_ = np.zeros(1+X.shape[1])
        #初始化一个成本向量，用来标记每次得到后的改进数值，能够判断每次改进的效果
        self.const_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            #output = w0 +w1*x1+...wn*xn
            errors = (y - output)
            #对X分量进行转质和点乘
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum()/2.0
            self.cost_.append(cost)
        return self

    #网络输入
    def net_input(self,X):
        return np.dot(X,self.w_[1:],self.w_[0])

    #激活函数
    def activation(self,X):
        return np.where(self.activation(X)>= 0,1,-1)


#
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

#

df = pd.read_csv(file,header=None)
#提取0～100行第四列数据 ,按行显示出来
X = df.iloc[0:100,[0,2]].values
y = df.loc[0:100,4].values
#print y
#转换当前类型
y = np.where(y == 'Iris-setosa',-1,1)
#调用自适应线性神经网络
ada = AdalineGD(eta=0.0001,n_iter=50)
ada.fit(X,y)
plot_decision_regions(X,y,classifier=ada)
plt.title('Adaline_Gradient descent')
plt.xlabel('花茎长度')
plt.ylabel('花瓣长度')
plt.legend(loc='uper left')
plt.show()

#展现学习过程中错误次数
plt.plot(range(1,len(ada.const_)+1,ada.const_,marker='o'))
plt.xlabel('Epoch')
plt.ylabel('sum-aquard-error')
plt.show()
