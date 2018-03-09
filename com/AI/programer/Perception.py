#!/usr/bin/python
# encoding=utf-8
import numpy as np
#自定义的感知器分类算法
class Perception(object):
    """
    eta : 学习率
    n_iter ：权重向量的训练次数
    w_ :神经分叉权重向量
    errors ：用于记录神经元判断出错次数
    """
    def __init__(self,eta = 0.01, n_iter = 10):
        self.eta = eta #学习率
        self.n_iter = n_iter #迭代次数
        pass
    def fit(self,x,y):
        """
        输入训练数据，培训神经元
        :param x: 输入样本向量 ：shape[n_samples,n_features] [[1,2,3],[4,5,6]]
        n_samples : 2 ,n_features :3
        :param y: 对应样本分类 :[1,-1]
        :return: 
        """
        #初始化权重向量为0，加1是因为前面算法提到的W0，也就是步调函数阈值
        self.w_ = np.zero(1+x.shape[1])
        self.errors = []

        #开始训练模型
        for _  in range (self.n_iter):
            errors = 0
            """
            X:[[1,2,3],[4,5,6]]
            y:[1,-1]
            zip(x,y) = [[1,2,3,1],[4,5,6,-1]]
            """
            for xi,target in zip (x,y):
                # updata=N *(y-y')
                update = self.eta * (target - self.predict(xi))
                """
                xi 是一个向量
                update * xi 等价于：
                [w1 = x[1]*update,w2 = x[2]*update,w3 = x[3]*update]
                """
                self.w_[0] += update * xi
                self.w_[1:] += update * xi
                errors += int(update !=0.0)
                self.errors_.append(errors)
                pass

            pass
        def net_input(self,x):
            """
            z=w0*1+w1*x1...+Wn*Xn
            :param self: 
            :param x: 
            :return: 
            """
            return np.dot(x,self.w_[1:])+self.w_[0]
            pass
        def predict(self,x):
            return np.where(self.net_input(x)>=0.0,1,-1)
            pass
        pass