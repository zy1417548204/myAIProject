#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from functools import wraps
import time
import tarfile
import scipy.io.wavfile as wav
import numpy as np
from core.calcmfcc import delta,calcMFCC
from six.moves import xrange as range
import os
import sys
import codecs
from keras.preprocessing.text import text_to_word_sequence, one_hot, Tokenizer;
from python_speech_features import mfcc
import pickle

#for i in np.arange(0, 3):
    #print i
#print s1#.max(0)
#print n
"""测试文本验证
"""
# s2 = """sgaggh 我们 是 朋友
# shhd 他们 不是 朋友"""
# texts =s2.split("\n")
# texts = [i.split(" ") for i in texts]
#
# all_words = [];   #这里面保存了所有的词，存在重复
# maxlen_char = 0;  #maxlen_char 是所有句子中字数最多
# labels_dict = {}  # 保存映射 ，当图片个数跟 句子个数不匹配的时候，以句子的标签为基准
# for i in np.arange(0, len(texts)):  #遍历每篇文本中所有句子
#     length = 0;
#     labels_dict[texts[i][0]] = i #{(第0句话的第一个词，0),(第1句话的第一个词，1)}
#     for j in texts[i][1:]:       #遍历每句话的所有词（除第一个词外）
#         length += len(j);  #统计一句话中的字数，注意不是词数
#     if maxlen_char <= length: maxlen_char = length;  #找出所有句子最多的字数
#     for j in np.arange(1, len(texts[i])): #遍历每句话的所有词（除第一个词外）
#         all_words.append(texts[i][j]);    #所有词
# #这个类允许向量化一个文本资料
# tok = Tokenizer(char_level=True);
# #根据文本列表更新内部词汇
# tok.fit_on_texts(all_words);
# char_index = tok.word_index; #字，索引 # vocab 全部转为了字编码，char_level=True这个参数很重要，如果为False，那么继续是词的编码
# index_char = dict((char_index[i], i) for i in char_index);  #编码 --> 字
# char_vec = np.zeros((len(texts), maxlen_char), dtype=np.float32)  # 句子 --> 编码向量
# char_length = np.zeros((len(texts), 1), dtype=np.float32)   # 句子中字数 #多少句 这个是一维的[5,7,8,...]
# for i in np.arange(0, len(texts)):
#     j = 0;
#     for i1 in texts[i][1:]: #每句话每个词
#         for ele in i1:      #每个字
#             char_vec[i, j] = char_index[ele]; #每句话每个字 标上索引
#             j += 1;
#     char_length[i] = j; # 句子中字数
# #index_char,char_index,char_length,char_vec,labels_dict
# print all_words
#
# print index_char
# print char_index
# print '\xe6\x88\x91\xe4\xbb\xac'
# i=0
# for o in char_vec:
#     s=''
#     print o[:int(char_length[i])]
#     for o1 in o[:int(char_length[i])]:
#        s+=index_char[o1]
#     print s
#     i+=1
# print char_vec
# print int(char_length[0])
# print labels_dict
"""测试语音映射
"""
#s=""
s=os.walk("/Users/edz/PycharmProjects/myAIProject/com/AI/MingGeTrain")

i=0
print s
for (dirpath_1, dirnames_1, s1) in s:  # dir S0002 ,S0003,...,S0723
    print s1
    # for one_dir in sorted(dirnames_1):  # 排序
    #     for (dirpath_2, _, filenames_2) in os.walk(os.path.join(dirpath_1, one_dir)):  # BAC009S0002W0122.wav,...
    #         for index, filename in enumerate(sorted(filenames_2)):
    #             print index