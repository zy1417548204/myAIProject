#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Compatibility imports
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
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

#解析元组
def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []
    for n, seq in enumerate(sequences):#枚举类[(0,12),(1,14)]
        #extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
        indices.extend(zip([n]*len(seq), range(len(seq))))#加入n=5,len(seq)=3格式为[[5 0][5 1][5 2]]
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64) #返回[[...,[5 0][5 1][5 2]],...]
    values = np.asarray(values, dtype=dtype)  #返回 [seq0,seq1,seq2]
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)
    return [indices, values, shape]


#解码文本
def txt_decode(label_file):
    with codecs.open(label_file, encoding="utf-8") as f:
        texts = f.read().split("\n"); #["我们 是 朋友","他们 不是 朋友", ...]
    del texts[-1] # 删除最后一行
    # print "texts len = %d" %len(texts)
    #print "texts[0] content: \n %s" %texts[0]
    texts = [i.split(" ") for i in texts] #把一句话拆分成词保存在texts中, [["我们","是"，"朋友"]，["他们"，"不是"，"朋友"]，...]
    all_words = [];   #这里面保存了所有的词，存在重复
    maxlen_char = 0;  #maxlen_char 是所有句子中字数最多
    labels_dict = {}  # 保存映射 ，当图片个数跟 句子个数不匹配的时候，以句子的标签为基准
    for i in np.arange(0, len(texts)):  #遍历每篇文本中所有句子
        length = 0;
        labels_dict[texts[i][0]] = i #{(第0句话的第一个词，0),(第1句话的第一个词，1)}
        for j in texts[i][1:]:       #遍历每句话的所有词（除第一个词外）
            length += len(j);  #统计一句话中的字数，注意不是词数
        if maxlen_char <= length: maxlen_char = length;  #找出所有句子最多的字数
        for j in np.arange(1, len(texts[i])): #遍历每句话的所有词（除第一个词外）
            all_words.append(texts[i][j]);    #所有词
    #这个类允许向量化一个文本资料
    tok = Tokenizer(char_level=True);
    #根据文本列表更新内部词汇
    tok.fit_on_texts(all_words);
    char_index = tok.word_index; #字，索引 # vocab 全部转为了字编码，char_level=True这个参数很重要，如果为False，那么继续是词的编码
    index_char = dict((char_index[i], i) for i in char_index);  #编码 --> 字
    char_vec = np.zeros((len(texts), maxlen_char), dtype=np.float32)  # 句子 --> 编码向量
    char_length = np.zeros((len(texts), 1), dtype=np.float32)   # 句子中字数 #多少句 这个是一维的[5,7,8,...]
    for i in np.arange(0, len(texts)):
        j = 0;
        for i1 in texts[i][1:]: #每句话每个词
            for ele in i1:      #每个字
                char_vec[i, j] = char_index[ele]; #每句话每个字 标上索引
                j += 1;
        char_length[i] = j; # 句子中字数
    #返回 索引字向量，字向量索引，每行字数，每个字索引的二维列表，{每句话的第一个词，第n句话索引}
    return index_char,char_index,char_length,char_vec,labels_dict#

#映射音频和文本文件
def map_wav_txt(img_path,char_index,char_length,char_vec,labels_dict,compute_sample_num,num_features,sample_rate,feature_mode):
    vggfeature_tensor = []  #
    labels_vec = []
    labels_length = []
    seq_length = []
    sample_index = 0
    if img_path:
        """os.walk(top, topdown=True, onerror=None, followlinks=False) 
        可以得到一个三元tupple(dirpath, dirnames, filenames), 
        第一个为起始路径，第二个为起始路径下的文件夹，第三个是起始路径下的文件。
        dirpath 是一个string，代表目录的路径，
        dirnames 是一个list，包含了dirpath下所有子目录的名字。
        filenames 是一个list，包含了非目录文件的名字。"""
        for (dirpath_1,dirnames_1,_) in os.walk(img_path):# dir S0002 ,S0003,...,S0723
            for one_dir in sorted(dirnames_1): #排序
                for (dirpath_2, _, filenames_2) in os.walk(os.path.join(dirpath_1,one_dir)):#BAC009S0002W0122.wav,...
                    for index, filename in enumerate(sorted(filenames_2)):
                        if filename.endswith("wav"):
                            wav_id = os.path.basename(filename).split('.')[0]#赋值wav_id=BAC009S0002W0122
                            if labels_dict.has_key(wav_id):
                                #labels_vec 每一行的每个字索引放进去
                                labels_vec.append(char_vec[labels_dict[wav_id]])  # labels_dict[wav_id] 保存的是序号,每句话的索引
                                #每句话的长度字数
                                labels_length.append(char_length[labels_dict[wav_id]])  # labels_dict[wav_id] 保存的是序号
                                #fs 表示采样率
                                fs, audio = wav.read(os.path.join(dirpath_2,filename))
                                if fs == sample_rate:
                                    None
                                elif fs == sample_rate * 2:
                                    audio = audio[::2]#隔一个取一个
                                else:
                                    print("sample rate is error!")
                                #inputs = mfcc(audio, samplerate=fs, numcep=num_features)
                                if feature_mode == "onlymfcc":
                                    # 2-D numpy array with shape (NUMFRAMES, features). Each frame containing feature_len of features.
                                    #返回一个2-Dnumpy 数组，包含特征长度的特征
                                    inputs = calcMFCC(audio, samplerate=fs, feature_len=num_features, mode="mfcc")
                                else:
                                    inputs = calcMFCC(audio, samplerate=fs, feature_len=num_features,mode=feature_mode)
                                    #Compute delta features from a feature vector sequence. 从特征序列向量中计算delta 特征
                                    #A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
                                    #返回一个包含delta 特征的numpy 数组，每一行有一个delta 特征向量
                                    delta_input = np.array(delta(inputs))
                                    delta_delta_input = np.array(delta(delta_input))
                                    #Join a sequence of arrays along an existing axi。 沿着一个存在的轴连接一个序列列表
                                    inputs = np.concatenate([inputs,delta_input,delta_delta_input], axis=1)
                                #序列长度
                                seq_length.append(inputs.shape[0])

                                train_inputs = np.asarray(inputs[np.newaxis, :])
                                #np.std 计算一个标准差在某个轴上
                                train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
                                vggfeature_tensor.append(train_inputs)
                                sample_index = sample_index + 1
                                if sample_index >= compute_sample_num:
                                    return vggfeature_tensor, np.array(labels_vec), np.array(labels_length), np.array(
                                    seq_length)
                        if sample_index % 100 == 0: print("Completed {}".format(str(sample_index * compute_sample_num ** -1)))
    #返回   训练数据 ，每个字的索引（二维），每句话字数长度，每个音频文件的序列长度
    return vggfeature_tensor,np.array(labels_vec),np.array(labels_length),np.array(seq_length)

#解压文件到指定目录
def unpack_file(wav_path):
    def unpack(filepath, target_dir, rm_tar=False):
        """Unpack the file to the target_dir."""
        print("Unpacking %s ..." % filepath)
        tar = tarfile.open(filepath)
        tar.extractall(target_dir)
        tar.close()
        if rm_tar == True:
            os.remove(filepath)
    for subfolder, _, filelist in sorted(os.walk(wav_path)): #返回一个新的存储列表
        for ftar in filelist:
            unpack(os.path.join(subfolder, ftar), subfolder, True)

#生成各种样本
def gen_all_kind_sample(num_features,batch_size,sample_rate,compute_sample_num):
    #判断特征个数
    if num_features == 13:
        feature_mode = "mfcc"
    elif num_features == 80:
        feature_mode = "fbank"
    elif num_features == 20:
        feature_mode = "onlymfcc"
    if compute_sample_num <= 1000:#计算样本量小于1000
        txt_path = "/mnt/steven/code/ctc_project/ctc_tensorflow_example/aishell_small.txt"
    else:
        txt_path = "/mnt/steven/data/data_aishell/transcript/aishell_transcript_v0.8.txt"
    wav_path = "/mnt/steven/data/data_aishell/wav"
    for _,_,filename in sorted(os.walk(wav_path)):
        if len(filename) != 0 and filename[0].endswith("tar.gz"):
            unpack_file(wav_path)
        else:
            None
    wav_path = os.path.join(wav_path,"train")
    index_char, char_index, char_length, char_vec, labels_dict = txt_decode(txt_path)
    print("aishell样本集中 取样了 %d 段音频,这些音频的字典中共有%d个字,其中最长的句子中包含 %d 个字" %(char_vec.shape[0],len(index_char.items()),char_vec.shape[1]))
    train_inputs, labels_vec, labels_length, train_seq_len = map_wav_txt(wav_path, char_index, char_length,
                                                                         char_vec, labels_dict,compute_sample_num,num_features,sample_rate,feature_mode)
    #每个音频句子最长
    max_seq_len = np.max(train_seq_len)
    #真实计算样本量
    real_compute_sample_num = len(train_inputs)

    if feature_mode == "onlymfcc":
        new_train = np.zeros([real_compute_sample_num, max_seq_len, num_features])
    else:
        new_train = np.zeros([real_compute_sample_num, max_seq_len, num_features*3])
    for i in range(real_compute_sample_num):
        new_train[i, :train_seq_len[i]] = train_inputs[i][0, :]
    print("label_vec_shape = %s,new_train_shape = %s, vocab len = %d" % (labels_vec.shape, new_train.shape,len(index_char)))
    save_path = os.path.join(os.getcwd(),"aishell_feature_%d_num_%d_samplerate_%d" % (num_features,real_compute_sample_num,sample_rate))
    print "save_path %s " % save_path
    if os.path.isdir(save_path):
        None
    else:
        os.mkdir(save_path)
    test_set_size = int(real_compute_sample_num * 0.1)
    np.savez(os.path.join(save_path,"train.npz"),
             train_x = new_train[:-test_set_size],  #语音数据序列             #new_train : 100 * 465 * 39
             train_y = labels_vec[:-test_set_size], #文本数据序列            #label_vec : 100 * 26
             train_seq_len = train_seq_len[:-test_set_size],     #seq_len   : 100 *  1
             vocab_char_index = char_index,                      #char_index: 1000 * 26
             vocab_index_char = index_char                       #index_char: 1000 * 26
             )
    np.savez(os.path.join(save_path,"validation.npz"),
             validation_x = new_train[-test_set_size:],
             validation_y = labels_vec[-test_set_size:],
             validation_seq_len = train_seq_len[-test_set_size:]
             )

if __name__ == '__main__':
    # 生成训练数据
    num_features = [13,20,80]# 80 * 3 fbank for seq2seq, 13 mfcc for ctc
    batch_size = 100
    # max sample num = 141600
    #录音时长：178小时. 录音人数：400人. 录音语言：中文. 录音地区：中国. 录音设备：高保真麦克风. 16000Hz，16bit.
    compute_sample_num = [100,400,1000,10000]
    sample_rate = [8000,16000]
    for num_feature in num_features:
        for oneitem in compute_sample_num:
            for onesamplerate in sample_rate:
                gen_all_kind_sample(num_feature,batch_size,onesamplerate,oneitem)
