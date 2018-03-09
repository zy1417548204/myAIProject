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

load_path = "tmpdata/aishell_feature_13_num_100_samplerate_8000"
train_ = np.load(os.path.join(load_path, "train.npz"))
train_inputs = train_["train_x"]
print train_inputs
train_targets = train_["train_y"]
train_seq_len = np.squeeze(train_["train_seq_len"].astype(np.int32))
char_index = train_["vocab_char_index"][()]
index_char = train_["vocab_index_char"][()]
train_num_examples = train_inputs.shape[0]