#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import os
################################################################################
from tensorflow.contrib.slim.python.slim.data.data_provider import DataProvider

"""1.生成tfrecord"""

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, labels, height, width, channels=3):
    """Build an Example proto for an example.

    Parameters
    ---------------
    @filename: string, path to an image file, e.g., '19901221.jpg'
    @image_buffer: string, JPEG encoding of RGB image
    @labels: list of semantic level name and one-hot encoder, 
        format list of tuple: [(level_name, [0,0,1,0,0,1,0,0]), ..]
    @height: integer, image height in pixels.
    @width: integer, image width in pixels.
    @channels: integer, image channels.
    Return:
    ---------------
    @example: Example proto
    """
    feature = {'image/height': _int64_feature(height),
               'image/width': _int64_feature(width),
               'image/channels': _int64_feature(channels),
               'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
               'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}
    for level_name, one_hot in labels:
        feature.update({'image/label/%s' % level_name: _int64_feature(one_hot)})
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example
# writer = tf.python_io.TFRecordWriter(output_file)
# example = balabala
# writer.write(example.SerializeToString())
#####################################################################################################
"""2.解析tfrecord"""


def parse_example_proto(example_serialized, semantic_level_settings):
    """Parses an Example proto containing a training example of an image.

    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields:
      image/class/root: [0,1,1,0,0,0]
      image/filename: '19901221.jpg'
      image/encoded: 

    Parameters
    ---------------
    @example_serialized: Scalar Tensor tf.string containing a serialized Example protocol buffer.
    @semantic_level_settings: Specify number of classes in each semantic level.
        format list of tuple: [(level_name, 10), ..]

    Return:
    ---------------
    @image_buffer: Tensor tf.string containing the contents of a JPEG file.
    @labels: List of Tensor tf.int32 containing the one-hot label.
    @filename: Tensor tf.string filename of the sample.
    """
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/filename': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    }
    for level, num in semantic_level_settings:
        feature_map.update({'image/label/%s' % level:
                                tf.FixedLenFeature([num], dtype=tf.int64, default_value=[-1] * num)})
    features = tf.parse_single_example(example_serialized, feature_map)
    labels = [tf.cast(features['image/label/%s' % level], dtype=tf.int32)
              for level, _ in semantic_level_settings]
    return features['image/encoded'], labels, features['image/filename']

################################################################################
"""二、QueueRunner"""
"""在使用文件对数据进行读取时，主要维护的两个队列。一个是文件的队列，
这个队列里面放的是TFrecord文件，另一个队列维护的是单个TFrecord中example的队列。 流程如下：

data_files = tf.gfile.Glob(tf_record_pattern)生成TFrecord文件列表，tf_record_pattern为正则表达式。生成列表也可以
使用其他的一些自定义方式例如os.system("find dir -name 'pattern'")方式。
filename_queue = tf.train.string_input_producer(data_files, shuffle=True, capacity=16)首先生成一个filename_queue,
这个队列中存放了所有的文件（包括文件的组织形式），通过shuffle参数指定
是不是打乱文件，一般在training的时候会设置成True，预测的时候设置成False，capacity为队列里至少的filename容量，这个参数随意。
使用tf.TFRecordReader()生成example的队列。

当reader的个数大于1的时候，使用QueueRunner进行调度，其中queue包括RandomShuffleQueue和FIFOQueue前者主要用在训练的时候，
后者主要用在预测的时候。其中dtype参数为[string]
启动多(单)线程读取数据。
获取一个batch的数据tf.train.batch_join。"""

def batch_inputs(dataset, batch_size, train, semantic_level_settings, num_preprocess_threads=16):
    """Generate batches of data for training or validating or something.

    Parameters
    ---------------
    @dataset: instance of Dataset class specifying the dataset.
    @batch_size: integer, number of examples in batch
    @train: The gotten batch data if for training or not.
        if True, number of reader if large and doesn't shuffle batch data
        if False, only 1 reader and data is not shuffled.
    @semantic_level_settings: Specify number of classes in each semantic level.
        format list of tuple: [(level_name, 10), ..]
    @num_preprocess_threads: integer, total number of preprocessing threads.

    Return:
    ---------------
    @images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                       image_size, 3].
    @labels: Dict, key is `level_name` in `semantic_level_settings`, value is label.
    @filenames: 1-D string Tensor of [batch_size].
    """
    # Force all input processing onto CPU in order to reserve the GPU for the forward and backward.
    with tf.device('/cpu:0'):
        with tf.name_scope('batch_processing'):
            data_files = dataset.data_files()
            if data_files is None:
                raise ValueError('No data files found for this dataset')

            examples_per_shard = 1024
            # Create filename_queue
            if train:
                filename_queue = tf.train.string_input_producer(data_files, shuffle=True, capacity=16)
                input_queue_memory_factor = 16
                num_readers = 4
            else:
                filename_queue = tf.train.string_input_producer(data_files, shuffle=False, capacity=1)
                input_queue_memory_factor = 1
                num_readers = 1
            if num_preprocess_threads % 4:
                raise ValueError('Please make num_preprocess_threads a multiple '
                                 'of 4 (%d % 4 != 0).', num_preprocess_threads)

            min_queue_examples = examples_per_shard * input_queue_memory_factor
            if train:
                examples_queue = tf.RandomShuffleQueue(
                    capacity=min_queue_examples + 3 * batch_size,
                    min_after_dequeue=min_queue_examples,
                    dtypes=[tf.string])
                # Create multiple readers to populate the queue of examples.
                enqueue_ops = []
                for _ in range(num_readers):
                    reader = dataset.reader()
                    _, value = reader.read(filename_queue)
                    enqueue_ops.append(examples_queue.enqueue([value]))

                tf.train.queue_runner.add_queue_runner(
                    tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
                example_serialized = examples_queue.dequeue()
            else:
                examples_queue = tf.FIFOQueue(
                    capacity=examples_per_shard + 3 * batch_size,
                    dtypes=[tf.string])
                # Create multiple readers to populate the queue of examples.
                reader = dataset.reader()
                _, example_serialized = reader.read(filename_queue)

            images_and_labels = []
            for thread_id in range(num_preprocess_threads):
                # Parse a serialized Example proto to extract the image and metadata.
                image_buffer, labels, filename = parse_example_proto(example_serialized,
                                                                     semantic_level_settings)
                image = decode_jpeg(image_buffer)
                if train:
                    image = distort_image(image, dataset.height, dataset.width, thread_id)
                else:
                    image = eval_image(image, dataset.height, dataset.width)

                # Finally, rescale to [-1,1] instead of [0, 1)
                image = tf.subtract(image, 0.5)
                image = tf.multiply(image, 2.0)
                images_and_labels.append([image, filename] + labels)

            batch_data = tf.train.batch_join(
                images_and_labels,
                batch_size=batch_size,
                capacity=2 * num_preprocess_threads * batch_size)

            # Get image data, filenames, level_labels separately.
            images = batch_data[0]
            images = tf.cast(images, tf.float32)
            images = tf.reshape(images, shape=[batch_size, dataset.height, dataset.width, 3])

            filenames = tf.reshape(batch_data[1], [batch_size])
            level_labels = {}
            for idx, settings in enumerate(semantic_level_settings):
                level_labels[settings[0]] = tf.reshape(batch_data[2 + idx], [batch_size, -1])

            return (images, level_labels, filenames)
################################################################################
#三、使用TFrecord和QueueRunner
images, labels, _ = DataProvider.batch_inputs(dataset, args.batch_size)
model = MultiLabelTree(images, labels, is_training=True, reuse=reuse)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=session, coord=coord)

TRAING
ITERS...

coord.request_stop()
coord.join(threads)
