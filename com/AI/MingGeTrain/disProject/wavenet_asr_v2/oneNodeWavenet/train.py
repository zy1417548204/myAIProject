#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/9/18 7:16 PM
# @Author  : renxiaoming@julive.com
# @Site    :
# @File    : train.py
# @Software: PyCharm

from __future__ import print_function

import datetime
import os

import numpy as np
import tensorflow as tf  # 1.0.0

from model import Model


def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return [indices, values, shape]

def count_params(model, mode='trainable'):
    ''' count all parameters of a tensorflow graph
    '''
    if mode == 'all':
        num = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in model.var_op])
    elif mode == 'trainable':
        num = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in model.var_trainable_op])
    else:
        raise TypeError('mode should be all or trainable.')
    print('number of '+mode+' parameters: '+str(num))
    return num


def train():
	# setting parameters
	batch_size = 64
	n_epoch = 3000

	keep = False

	# load speech data
	real_compute_sample_num = 100
	sample_rate = 16000
	num_features = 20
	if num_features == 20:
		n_mfcc = num_features
	else:
		n_mfcc = 3 * num_features


	data_path = "/root/mickey21/wavenet"
	load_path = os.path.join(data_path, "aishell_feature_%d_num_%d_samplerate_%d" % (
	num_features, real_compute_sample_num, sample_rate))
	print("load_path %s" % load_path)
	train_ = np.load(os.path.join(load_path,"train.npz"))
	train_inputs = train_["train_x"]
	labels_vec = train_["train_y"]
	train_seq_len = train_["train_seq_len"]
	char_index = train_["vocab_char_index"][()]
	index_char = train_["vocab_index_char"][()]


	train_num_examples = train_inputs.shape[0]
	train_num_batches_per_epoch = int(train_num_examples / batch_size)

	train_targets = []
	for index in range(labels_vec.shape[0] // batch_size):
		train_targets.append(sparse_tuple_from(labels_vec[index * batch_size: (index + 1) * batch_size, :]))

	# 设置验证集合
	validation_ = np.load(os.path.join(load_path,"validation.npz"))
	val_inputs = validation_["validation_x"]
	val_label_vec = validation_["validation_y"]
	val_seq_len = validation_["validation_seq_len"]
	val_num_examples = val_inputs.shape[0]
	val_num_batches_per_epoch = int(val_num_examples / batch_size)
	val_targets = []
	for index in range(val_label_vec.shape[0] // batch_size):
		val_targets.append(sparse_tuple_from(val_label_vec[index * batch_size: (index + 1) * batch_size, :]))

	#val_inputs, val_targets, val_seq_len = train_inputs,train_targets,  train_seq_len

	#
	model = Model(vocab_size = len(index_char.items()),max_seq_len = np.max(train_seq_len),
				  batch_size=batch_size, n_mfcc=n_mfcc)
	num_params = count_params(model, mode='trainable')
	print("num_params = %d " % num_params)

	#saver = tf.train.Saver(tf.global_variables())
	saver = tf.train.Saver()
	load_path = os.path.join(os.getcwd(), "model")

	with tf.Session() as sess:
		if keep:
			print('Model restored from:' + load_path)
			saver.restore(sess, tf.train.latest_checkpoint(load_path))
		else:
			print('Initializing')
			tf.global_variables_initializer().run()

		train_cost = train_ler = 0

		for cur_epoch in range(n_epoch):
			start = datetime.datetime.now();print ("training start @ start")
            #print ("training start @ start")
			for batch in range(train_num_batches_per_epoch):

				feed = {model.input_data: train_inputs[batch * batch_size : (batch + 1) * batch_size],
						model.targets: train_targets[batch],
						model.seq_len:train_seq_len[batch * batch_size : (batch + 1) * batch_size]}
				batch_ler ,batch_cost, _ = sess.run([model.ler,model.cost, model.optimizer_op], feed_dict=feed)

				train_cost += batch_cost * batch_size
				train_ler += batch_ler * batch_size


			sum_val_cost = sum_val_ler = 0.
			for batch in range(val_num_batches_per_epoch):
				val_feed = {model.input_data: val_inputs[batch * batch_size: (batch + 1) * batch_size],
							model.targets: val_targets[batch],
							model.seq_len: val_seq_len[batch * batch_size: (batch + 1) * batch_size]}
				# run_sense = session.run([dense],feed)
				val_cost, val_ler = sess.run([model.cost, model.ler], feed_dict=val_feed)
				sum_val_cost += val_cost * batch_size
				sum_val_ler += val_ler * batch_size

				#print("epoch: %d/%d, batch: %d/%d, loss: %s, time: %.3f."%(epoch, n_epoch, batch, num_batches_per_epoch, train_loss, end-start))
			train_cost /= train_num_examples
			train_ler /= train_num_examples
			sum_val_cost /= val_num_examples
			sum_val_ler /= val_num_examples
			log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
			print(log.format(cur_epoch + 1, n_epoch, train_cost, train_ler,sum_val_cost,sum_val_ler, datetime.datetime.now()- start))

			# save models
			if cur_epoch % 5 ==0:
				saver.save(sess, os.path.join(os.getcwd(), 'model','speech.module'), global_step=cur_epoch)


if __name__ == '__main__':
	train()