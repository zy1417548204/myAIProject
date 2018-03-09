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
from collections import Counter
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
def txt_decode(label_file):
    with codecs.open(label_file, encoding="utf-8") as f:
        texts = f.read().split("\n"); #["我们 是 朋友","他们 不是 朋友", ...]
    del texts[-1]
    # print "texts len = %d" %len(texts)
    #print "texts[0] content: \n %s" %texts[0]
    texts = [i.split(" ") for i in texts] #把一句话拆分成词保存在texts中, [["我们","是"，"朋友"]，["他们"，"不是"，"朋友"]，...]
    all_words = [];   #这里面保存了所有的词，存在重复
    maxlen_char = 0;  #maxlen_char 是所有句子中字数最多
    labels_dict = {}  # 保存映射 ，当图片个数跟 句子个数不匹配的时候，以句子的标签为基准
    for i in np.arange(0, len(texts)):
        length = 0;
        labels_dict[texts[i][0]] = i
        for j in texts[i][1:]:
            length += len(j);  #统计一句话中的字数，注意不是词数
        if maxlen_char <= length: maxlen_char = length;
        for j in np.arange(1, len(texts[i])):
            all_words.append(texts[i][j]);
    tok = Tokenizer(char_level=True);
    tok.fit_on_texts(all_words);
    char_index = tok.word_index;  # vocab 全部转为了字编码，char_level=True这个参数很重要，如果为False，那么继续是词的编码
    index_char = dict((char_index[i], i) for i in char_index);  #编码 --> 字
    char_vec = np.zeros((len(texts), maxlen_char), dtype=np.float32)  # 句子 --> 编码向量
    char_length = np.zeros((len(texts), 1), dtype=np.float32)   # 句子中字数
    for i in np.arange(0, len(texts)):
        j = 0;
        for i1 in texts[i][1:]:
            for ele in i1:
                char_vec[i, j] = char_index[ele];
                j += 1;
        char_length[i] = j;
    return index_char,char_index,char_length,char_vec,labels_dict
def map_wav_txt(img_path,char_index,char_length,char_vec,labels_dict,compute_sample_num,num_features,sample_rate,feature_mode):
    vggfeature_tensor = []
    labels_vec = []
    labels_length = []
    seq_length = []
    sample_index = 0
    if img_path:
        for (dirpath_1,dirnames_1,_) in os.walk(img_path):# dir S0002 ,S0003,...,S0723
            for one_dir in sorted(dirnames_1):
                for (dirpath_2, _, filenames_2) in os.walk(os.path.join(dirpath_1,one_dir)):#BAC009S0002W0122.wav,...
                    for index, filename in enumerate(sorted(filenames_2)):
                        if filename.endswith("wav"):
                            wav_id = os.path.basename(filename).split('.')[0]
                            if labels_dict.has_key(wav_id):
                                labels_vec.append(char_vec[labels_dict[wav_id]])  # labels_dict[wav_id] 保存的是序号
                                labels_length.append(char_length[labels_dict[wav_id]])  # labels_dict[wav_id] 保存的是序号
                                fs, audio = wav.read(os.path.join(dirpath_2,filename))
                                if fs == sample_rate:
                                    None
                                elif fs == sample_rate * 2:
                                    audio = audio[::2]
                                else:
                                    print("sample rate is error!")
                                #inputs = mfcc(audio, samplerate=fs, numcep=num_features)
                                if feature_mode == "onlymfcc":
                                    inputs = calcMFCC(audio, samplerate=fs, feature_len=num_features, mode="mfcc")
                                else:
                                    inputs = calcMFCC(audio, samplerate=fs, feature_len=num_features,mode=feature_mode)
                                    delta_input = np.array(delta(inputs))
                                    delta_delta_input = np.array(delta(delta_input))
                                    inputs = np.concatenate([inputs,delta_input,delta_delta_input], axis=1)
                                seq_length.append(inputs.shape[0])
                                train_inputs = np.asarray(inputs[np.newaxis, :])
                                train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
                                vggfeature_tensor.append(train_inputs)
                                sample_index = sample_index + 1
                                if sample_index >= compute_sample_num:
                                    return vggfeature_tensor, np.array(labels_vec), np.array(labels_length), np.array(
                                    seq_length)
                        if sample_index % 100 == 0: print("Completed {}".format(str(sample_index * compute_sample_num ** -1)))
    return vggfeature_tensor,np.array(labels_vec),np.array(labels_length),np.array(seq_length)
def unpack_file(wav_path):
    def unpack(filepath, target_dir, rm_tar=False):
        """Unpack the file to the target_dir."""
        print("Unpacking %s ..." % filepath)
        tar = tarfile.open(filepath)
        tar.extractall(target_dir)
        tar.close()
        if rm_tar == True:
            os.remove(filepath)
    for subfolder, _, filelist in sorted(os.walk(wav_path)):
        for ftar in filelist:
            unpack(os.path.join(subfolder, ftar), subfolder, True)
def gen_all_kind_sample(num_features,batch_size,sample_rate,compute_sample_num):
    if num_features == 13:
        feature_mode = "mfcc"
    elif num_features == 80:
        feature_mode = "fbank"
    elif num_features == 20:
        feature_mode = "onlymfcc"
    if compute_sample_num <= 1000:
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
    train_inputs, labels_vec, labels_length, train_seq_len = map_wav_txt(wav_path, char_index, char_length,
                                                                         char_vec, labels_dict,compute_sample_num,num_features,sample_rate,feature_mode)
    max_seq_len = np.max(train_seq_len)
    if feature_mode == "fbank":
        max_seq_len = int(np.ceil(max_seq_len/8.0)) * 8
    real_compute_sample_num = len(train_inputs)
    if feature_mode == "onlymfcc":
        new_train = np.zeros([real_compute_sample_num, max_seq_len, num_features])
    else:
        new_train = np.zeros([real_compute_sample_num, max_seq_len, num_features*3])
    for i in range(real_compute_sample_num):
        new_train[i, :train_seq_len[i]] = train_inputs[i][0, :]
    print("aishell样本集中 取样了 %d 段音频,这些音频的字典中共有%d个字,其中最长的句子中包含 %d 个字" %(char_vec.shape[0],len(index_char.items()),char_vec.shape[1]))
    print("label_vec_shape = %s,new_train_shape = %s, vocab len = %d" % (labels_vec.shape, new_train.shape,len(index_char)))
    save_path = os.path.join(os.getcwd(),"aishell_feature_%d_num_%d_samplerate_%d" % (num_features,real_compute_sample_num,sample_rate))
    print "save_path %s " % save_path
    if os.path.isdir(save_path):
        None
    else:
        os.mkdir(save_path)
    test_set_size = int(real_compute_sample_num * 0.1)
    np.savez(os.path.join(save_path,"train.npz"),
             train_x = new_train[:-test_set_size],               #new_train : 100 * 465 * 39
             train_y = labels_vec[:-test_set_size],              #label_vec : 100 * 26
             train_seq_len = train_seq_len[:-test_set_size],     #seq_len   : 100 *  1
             vocab_char_index = char_index,                      #char_index: 1000 * 26
             vocab_index_char = index_char                       #index_char: 1000 * 26
             )
    np.savez(os.path.join(save_path,"validation.npz"),
             validation_x = new_train[-test_set_size:],
             validation_y = labels_vec[-test_set_size:],
             validation_seq_len = train_seq_len[-test_set_size:]
             )
def fbank_forseq2seq(num_features,batch_size,sample_rate,compute_sample_num):
    def _build_vocab(text_path, threshold=1):
        with codecs.open(text_path, encoding="utf-8") as f:
            texts = f.read().split("\n");  # ["?? ? ??","?? ?? ??", ...]
        del texts[-1]
        text_dict = {}
        counter = Counter()
        max_text_len = 0
        for i in texts[:compute_sample_num]:
            index, value = i.split(" ", 1)
            words = value.replace(' ', '')
            for w in words:
                counter[w] += 1
            text_dict[index] = words
            if len(words) > max_text_len:
                max_text_len = len(words)
        vocab = [word for word in counter if counter[word] >= threshold]
        print ('Filtered %d words to %d words with word count threshold %d.' % (
        len(counter), len(vocab), threshold))
        word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
        idx = 3
        for word in vocab:
            word_to_idx[word] = idx
            idx += 1
        return word_to_idx,text_dict,max_text_len
    def _build_caption_vector(text_dict, word_to_idx, max_length=15):
        n_examples = len(text_dict)
        captions = np.ndarray((n_examples, max_length + 2)).astype(np.int32)
        captions_wav_names = []
        for i, img_path in enumerate(text_dict.iterkeys()):
            captions_wav_names.append(img_path)
            caption = text_dict[img_path]
            words = caption.replace(' ', '')  # caption contrains only lower-case words
            cap_vec = []
            cap_vec.append(word_to_idx['<START>'])
            for word in words:
                if word in word_to_idx:
                    cap_vec.append(word_to_idx[word])
            cap_vec.append(word_to_idx['<END>'])
            # pad short caption with the special null token '<NULL>' to make it fixed-size vector
            if len(cap_vec) < (max_length + 2):
                for j in range(max_length + 2 - len(cap_vec)):
                    cap_vec.append(word_to_idx['<NULL>'])
            captions[i, :] = np.asarray(cap_vec)
        # print "Finished building caption vectors"
        return captions, captions_wav_names
    def read_wav(img_path,wav_names):
        feature = []
        seq_length = []
        for wav_name in wav_names:
            complete_filename = os.path.join(img_path, wav_name[6:11], wav_name + ".wav")
            if os.path.exists(complete_filename):
                fs, audio = wav.read(complete_filename)
                if fs == sample_rate:
                    None
                elif fs == sample_rate * 2:
                    audio = audio[::2]
                else:
                    print("sample rate is error!")
                if feature_mode == "onlymfcc":
                    inputs = calcMFCC(audio, samplerate=fs, feature_len=num_features, mode="mfcc")
                else:
                    inputs = calcMFCC(audio, samplerate=fs, feature_len=num_features, mode=feature_mode)
                    delta_input = np.array(delta(inputs))
                    delta_delta_input = np.array(delta(delta_input))
                    inputs = np.concatenate([inputs, delta_input, delta_delta_input], axis=1)
                seq_length.append(inputs.shape[0])
                feature.append(inputs)
        new_train = np.zeros([len(feature), np.max(seq_length), inputs.shape[1]])
        for i in range(len(feature)):
            new_train[i, :seq_length[i]] = feature[i]
        return np.array(new_train),  np.array(seq_length)[:,np.newaxis]
    if num_features == 13:
        feature_mode = "mfcc"
    elif num_features == 80:
        feature_mode = "fbank"
    elif num_features == 20:
        feature_mode = "onlymfcc"
    if compute_sample_num <= 1000:
        txt_path = "/mnt/steven/code/ctc_project/ctc_tensorflow_example/aishell_small.txt"
    else:
        txt_path = "/mnt/steven/data/data_aishell/transcript/aishell_transcript_v0.8.txt"
    word_to_idx,text_dict,max_text_len = _build_vocab(text_path=txt_path)
    captions_vec, captions_wav_names = _build_caption_vector(text_dict, word_to_idx, max_length=max_text_len)
    wav_path = "/mnt/steven/data/data_aishell/wav"
    wav_path = os.path.join(wav_path, "train")
    feature,seq_length = read_wav(img_path = wav_path,wav_names = captions_wav_names)
    save_path = os.path.join(os.getcwd(),"aishell_feature_%d_num_%d_samplerate_%d" % (num_features,compute_sample_num,sample_rate))
    print "save_path %s " % save_path
    if os.path.isdir(save_path):
        None
    else:
        os.mkdir(save_path)
    test_set_size = int(compute_sample_num * 0.01)
    np.savez(os.path.join(save_path,"train.npz"),
             train_x = feature[:-test_set_size],               #new_train : 100 * 465 * 39
             train_y = captions_vec[:-test_set_size],              #label_vec : 100 * 26
             train_seq_len = seq_length[:-test_set_size],     #seq_len   : 100 *  1
             vocab_char_index = word_to_idx,                      #char_index: 1000 * 26
             vocab_index_char = {i: w for w, i in word_to_idx.iteritems()}                       #index_char: 1000 * 26
             )
    np.savez(os.path.join(save_path,"validation.npz"),
             validation_x = feature[-test_set_size:],
             validation_y = captions_vec[-test_set_size:],
             validation_seq_len = seq_length[-test_set_size:]
             )
if __name__ == '__main__':
    # 生成训练数据
    num_features = [80]# 80 * 3 fbank for seq2seq, 13 mfcc for ctc
    batch_size = 100
    # max sample num = 141600
    #录音时长：178小时. 录音人数：400人. 录音语言：中文. 录音地区：中国. 录音设备：高保真麦克风. 16000Hz，16bit.
    compute_sample_num = [100,400,1000,10000]
    sample_rate = [8000,16000]
    for num_feature in num_features:
        for oneitem in compute_sample_num:
            for onesamplerate in sample_rate:
                fbank_forseq2seq(num_feature,batch_size,onesamplerate,oneitem)
                #gen_all_kind_sample(num_feature,batch_size,onesamplerate,oneitem)
