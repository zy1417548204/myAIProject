#!/usr/bin/python


import tensorflow as tf
#from PIL.ImageCms import FLAGS
from tensorflow.contrib.learn.python.learn.datasets import mnist
from tensorflow.contrib.learn.python.learn.utils.inspect_checkpoint import FLAGS



def main(unused_argv):
    #获取数据
    data_sets = mnist.read_data_sets(FLAGS.directory,
                                     dtype=tf.uint8,  #注意这里的编码是uint8
                                     reshape=False,
                                     validation_size=FLAGS.validation_size)
    pass