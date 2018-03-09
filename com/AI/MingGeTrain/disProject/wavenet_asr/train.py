#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/9/18 7:16 PM
# @Author  : renxiaoming@julive.com
# @Site    :
# @File    : train.py
# @Software: PyCharm

from __future__ import print_function
from utils import SpeechLoader
import pickle
import numpy as np
from model import Model
import tensorflow as tf  # 1.0.0
import time
import os
import datetime

#将日志输出到某个文件
def save(filename, contents):
  fh = open(filename, 'a')#w:写
  fh.write(contents)
  fh.close()
#解析元祖，传入sequences，np数组
def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []

    for n, seq in enumerate(sequences):#强转为枚举类
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return [indices, values, shape]

#统计参数
def count_params(model, mode='trainable'):
    ''' count all parameters of a tensorflow graph
    '''
    if mode == 'all':
        num = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in model.var_op])
    elif mode == 'trainable':
        num = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in model.var_trainable_op])
    else:
        raise TypeError('model should be all or trainable.')
    print('number of ' + mode + ' parameters: ' + str(num))
    return num

""" 批次大小50，训练次数200轮"""
def train():
    # setting parameters
    batch_size = 50
    n_epoch = 200
    #梅尔频率倒谱系数（语音识别）
    n_mfcc = 20

    # load speech data
    #计算样本数
    compute_sample_num = 20  #aishell_feature_20_num_100_samplerate_16000
    #采样率
    sample_rate = 16000
    #数据路径
    data_path = "/root/mickey21/wavenet"
    #join（）连接两个路径或这更多的路径
    print ("load data start ")
    load_path = os.path.join(data_path, "aishell_feature_%d_num_100_samplerate_%d" % (compute_sample_num, sample_rate))

    #
    new_train = np.load(os.path.join(load_path, "wav_feature.npy"))
    labels_vec = np.load(os.path.join(load_path, "txt_decoder.npy"))
    train_seq_len = np.load(os.path.join(load_path, "wav_seqlen.npy"))
    #pickle对象持久化的一种方式
    char_index = pickle.load(open(os.path.join(load_path, 'char_index.pkl'), 'rb'))
    index_char = pickle.load(open(os.path.join(load_path, 'index_char.pkl'), 'rb'))
    print("load data over ")
    num_examples = new_train.shape[0]
    #每一轮批次数
    num_batches_per_epoch = int(num_examples / batch_size)

    train_inputs = new_train
    train_targets = []
    for index in range(labels_vec.shape[0] // batch_size):
        train_targets.append(sparse_tuple_from(labels_vec[index * batch_size: (index + 1) * batch_size, :]))

    # 设置验证集合
    val_inputs, val_targets, val_seq_len = train_inputs, train_targets, train_seq_len

    #定义网络模型
    model = Model(vocab_size=len(index_char.items()), max_seq_len=np.max(train_seq_len),
                  batch_size=batch_size, n_mfcc=n_mfcc)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())
        train_cost = train_ler = 0
        log="log"
        for cur_epoch in range(n_epoch):
            start = datetime.datetime.now()
            save(log,"第"+str(cur_epoch)+"开始于"+str(start)+"""
            """)
            print ("第"+str(cur_epoch)+"开始于"+str(start))
            for batch in range(num_batches_per_epoch):
                feed = {model.input_data: train_inputs[batch * batch_size: (batch + 1) * batch_size],
                        model.targets: train_targets[batch],
                        model.seq_len: train_seq_len[batch * batch_size: (batch + 1) * batch_size]}
                batch_cost, _ = sess.run([model.cost, model.optimizer_op], feed_dict=feed)

                train_cost += batch_cost * batch_size
                train_ler += sess.run(model.ler, feed_dict=feed) * batch_size

            sum_val_cost = sum_val_ler = 0.
            # for batch in range(num_batches_per_epoch):
            # 	val_feed = {model.input_data: val_inputs[batch * batch_size: (batch + 1) * batch_size],
            # 				model.targets: val_targets[batch],
            # 				model.seq_len: val_seq_len[batch * batch_size: (batch + 1) * batch_size]}
            # 	# run_sense = session.run([dense],feed)
            # 	val_cost, val_ler = sess.run([model.cost, model.ler], feed_dict=val_feed)
            # 	sum_val_cost += val_cost
            # 	sum_val_ler += val_ler

            # print("epoch: %d/%d, batch: %d/%d, loss: %s, time: %.3f."%(epoch, n_epoch, batch, num_batches_per_epoch, train_loss, end-start))
            train_cost /= num_examples
            train_ler /= num_examples
            log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
            print(log.format(cur_epoch + 1, n_epoch, train_cost, train_ler, sum_val_cost / num_batches_per_epoch,
                             sum_val_ler / num_batches_per_epoch, datetime.datetime.now() - start))
            save(log, "第" + str(cur_epoch) + "结束于耗时" + str(datetime.datetime.now() - start) + """
                        """)
            # save models  os.getcwd()返回一个当前的工作目录
            if cur_epoch % 5 == 0:
                saver.save(sess, os.path.join(os.getcwd(), 'model', 'speech.module'), global_step=cur_epoch)


if __name__ == '__main__':
    train()