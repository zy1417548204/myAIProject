# coding=utf-8
import tempfile

import numpy as np
import tensorflow as tf
import os
import math
import time
from tensorflow.examples.tutorials.mnist import input_data
import pickle
from model import Model
import datetime

"""
分布式布置Wavenet
"""


#将日志输出到某个文件
# def save(filename, contents):
#   fh = open(filename, 'a')#w:写
#   fh.write(contents)
#   fh.close()

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

"""本代码采用图间模式，异步更新"""
#忽视忽视警告，并屏蔽警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#定义一些常量，用于构建数据流图
flags = tf.app.flags
flags.DEFINE_string("data_dir","/tmp/mnist-data","Directory for storing mnist data")
#只下载数据，不做其他操作
flags.DEFINE_boolean("download_only",False,"Only perform downloading of data;Do not to"
                                           "sessioin preparation,model definition or training")
#task_index 从0开始。0代表用来厨师化变量第一个任务
flags.DEFINE_integer("task_index",None,"Worker task index,should >=0.task_index=0 is the master"
                                       " worker task the performs the variable initialization")
#每台机器第GPU个数，这里在集群上跑，"""没GPU"""
flags.DEFINE_integer("num_gpus",0,"Total number of gpus for each machine.If you dont use GPU,please it to '0'")
#在同步训练模式下，设置收集的工作节点的数量。默认是工作的总数
flags.DEFINE_integer("replicas_to_aggregate",
                     None,"Number of replicas to aggregate before parameter update is applied"
                          "(For sync_replicas mode only;default:num workers)")
#梅尔频率倒谱系数（语音识别）
flags.DEFINE_integer("n_mfcc",20,"Number of units in the hidden layer of the NN")
# 训练次数
flags.DEFINE_integer("n_epoch",None, "Number of (gloabal) training steps to perform")
# cpu索引
flags.DEFINE_integer("cpu_index",None, "Number of (gloabal) training steps to perform")
#每一批次样本数
flags.DEFINE_integer("batch_size",50,"Training batch size")
#计算样本数
flags.DEFINE_integer("compute_sample_num",10000,"Training batch size")
#采样率
flags.DEFINE_integer("sample_rate",16000,"Training batch size")
#学习率
#flags.DEFINE_float("learning_rate",0.01,"Learning rate")
#使用同步训练／异步训练
flags.DEFINE_boolean("sync_replicas",False,"Use the sync_replicas (sysnchronized replicas)"
                                           " mode,wherein the parameter updates from workers are "
                                           "aggregated before applied to avoid stale gradients")
#如果服务器已经存在，采用gRPC协议通信；如果不存在采用进程间通信
flags.DEFINE_boolean("existing_servers",False,"Whether servers already exists.If True,will use "
                                              "the worker hosts via their GRPC URLS(one client "
                                              "process per worker hosts).Otherwise,will create an in_process Tensorflow server.")
#参数服务器主机
flags.DEFINE_string("ps_hosts","47.95.32.212:2222","Comma_separated list of hostname:port pairs")
#工作节点主机
"""
47.95.32.212 iZ2zedu05kkqson2296xbcZ    作为参数服务器
47.95.32.225 iZ2zedu05kkqson2296xb8Z
47.95.33.5   iZ2zedu05kkqson2296xbaZ #
47.94.41.45  iZ2zedu05kkqson2296xbbZ # 无hbase
47.95.33.8  iZ2zedu05kkqson2296xbdZ # 无hbase @有presto
47.95.33.15  iZ2zedu05kkqson2296xb9Z # 无hbase @有presto
47.94.40.26  iZ2zedu05kkqson2296xbeZ # 无hbase
47.95.32.137  iZ2zedu05kkqson2296xbiZ # @有presto
47.93.136.26  iZ2zedu05kkqson2296xbgZ # 
47.95.32.230  iZ2zedu05kkqson2296xbfZ # 无hbase@有presto
47.95.32.182  iZ2zedu05kkqson2296xbhZ # 无hbase@有presto
47.93.55.184  gpu 
"""
flags.DEFINE_string("worker_hosts","47.95.32.212:2223,47.95.32.212:2224,47.95.32.212:2225,47.95.32.212:2226,47.95.32.212:2227,"
                                   "47.95.32.225:2222,47.95.32.225:2223,47.95.32.225:2224,47.95.32.225:2225,47.95.32.225:2226,"
                                   "47.95.33.5:2222,47.95.33.5:2223,47.95.33.5:2224,47.95.33.5:2225,47.95.33.5:2226,"
                                   "47.94.41.45:2222,47.94.41.45:2223,47.94.41.45:2224,47.94.41.45:2225,47.94.41.45:2226,"
                                   "47.95.33.8:2222,47.95.33.8:2223,47.95.33.8:2224,47.95.33.8:2225,47.95.33.8:2226,"
                                   "47.95.33.15:2222,47.95.33.15:2223,47.95.33.15:2224,47.95.33.15:2225,47.95.33.15:2226,"
                                   "47.94.40.26:2222,47.94.40.26:2223,47.94.40.26:2224,47.94.40.26:2225,47.94.40.26:2226,"
                                   "47.95.32.137:2222,47.95.32.137:2223,47.95.32.137:2224,47.95.32.137:2225,47.95.32.137:2226,"
                                   "47.93.136.26:2222,47.93.136.26:2223,47.93.136.26:2224,47.93.136.26:2225,47.93.136.26:2226,"
                                   "47.95.32.230:2222,47.95.32.230:2223,47.95.32.230:2224,47.95.32.230:2225,47.95.32.230:2226,"
                                   "47.95.32.182:2222,47.95.32.182:2223,47.95.32.182:2224,47.95.32.182:2225,47.95.32.182:2226",
                    "Comma_separated list of hostname :port pairs")
#本作业是工作节点还是参数服务器
flags.DEFINE_string("job_name",None,"job name:worker or ps")
FLAGS=flags.FLAGS
#IMAGE_PIXELS=28

#读取集群描述信息
ps_spec=FLAGS.ps_hosts.split(",")
worker_spec=FLAGS.worker_hosts.split(",")
"""临时添加"""
num_workers=len(worker_spec)
#创建TensorFlow集群描述对象
cluster=tf.train.ClusterSpec({
    "ps":ps_spec,
    "worker":worker_spec
})

#为本地执行的任务创建TensorFlow serverdui象
if not  FLAGS.existing_servers:
    """
    创建本地Server 对象，从tf.train.Server 这个定义开始，每个节点开始不同
    根据执行的命令的参数（作业名字）不同，决定啦这个任务是那个任务
    如果作业名字是ps，进程就加入到这里，作为参数更新的服务，等待其他工作节点给他提交参数更新的数据
    如果作业名字是worker，就执行后面的计算任务
    """
    server=tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)

    #如果是参数服务器，直接启动即可。这时，进程就会阻塞在这里
    #下面的tf.train.replica_device_setter代码会将参数指定给ps_server保管
    if FLAGS.job_name =="ps":
        server.join()

#找出worker的主节点，即task_index为0的点
is_chief=(FLAGS.task_index==0)
#如果使用GPU
if FLAGS.num_gpus >0:
    # if FLAGS.num_gpus < num_workers:
    #     raise ValueError("number of gpus is less than number of workers")
    gpu=(FLAGS.task_index %FLAGS.num_gpus)
    #分配worker到指定到gpu上运行
    worker_device ="/job:worker/task:%d/gpu:%d"%(FLAGS.task_index,gpu)
elif FLAGS.num_gpus ==0:
    #把CPU分配给worker
    #cpu = 0
    worker_device ="/job:worker/task:%d/cpu:%d"%(FLAGS.task_index,FLAGS.cpu_index)
#在这个with 语句之下定义的参数，会自动分配到参数服务器上去定义，如果有多个参数服务器就轮流分配
with tf.device(
    tf.train.replica_device_setter(
        worker_device=worker_device,
        ps_device="/job:ps/cpu:0",
        cluster=cluster
    )
):
    # 数据路径
    data_path = "/root/mickey21/wavenet/"
    compute_sample_num,sample_rate=FLAGS.compute_sample_num,FLAGS.sample_rate
    batch_size,n_mfcc=FLAGS.batch_size,FLAGS.n_mfcc
    n_epoch=FLAGS.n_epoch

    # join（）连接两个路径或这更多的路径
    load_path = os.path.join(data_path, "aishell_num_%d_samplerate_%d" % (compute_sample_num, sample_rate))

    # 定义全局步长，默认值为0
    global_step = tf.Variable(0, name="global_step", trainable=False)
    print("start loading data ")
    new_train = np.load(os.path.join(load_path, "wav_feature.npy"))
    labels_vec = np.load(os.path.join(load_path, "txt_decoder.npy"))
    train_seq_len = np.load(os.path.join(load_path, "wav_seqlen.npy"))
    # pickle对象持久化的一种方式
    char_index = pickle.load(open(os.path.join(load_path, 'char_index.pkl'), 'rb'))
    index_char = pickle.load(open(os.path.join(load_path, 'index_char.pkl'), 'rb'))
    print("loading data over")
    num_examples = new_train.shape[0]

    # 每一轮批次数
    num_batches_per_epoch = int(num_examples / batch_size)

    train_inputs = new_train
    train_targets = []
    for index in range(labels_vec.shape[0] // batch_size):
        train_targets.append(sparse_tuple_from(labels_vec[index * batch_size: (index + 1) * batch_size, :]))

    # 设置验证集合
    val_inputs, val_targets, val_seq_len = train_inputs, train_targets, train_seq_len

    # 定义网络模型
    model = Model(vocab_size=len(index_char.items()), max_seq_len=np.max(train_seq_len),
                  batch_size=batch_size, n_mfcc=n_mfcc)
    #定义损失函数和值
    loss = tf.nn.ctc_loss(model.targets, model.logit, model.seq_len, time_major=False)
    cost = tf.reduce_mean(loss)
    #定义优化器
    optimizer = tf.train.AdamOptimizer()
    var_list = [var for var in tf.trainable_variables()]
    gradient = optimizer.compute_gradients(cost, var_list=var_list)
    #optimizer_op = optimizer.apply_gradients(gradient)

    #转质
    decoded = tf.transpose(model.logit, perm=[1, 0, 2])
    decoded, log_prob = tf.nn.ctc_greedy_decoder(decoded, model.seq_len)
    #
    if FLAGS.sync_replicas:
        # 同步模式计算更新梯度
        rep_op = tf.train.SyncReplicasOptimizer(optimizer,
                                                replicas_to_aggregate=len(
                                                    worker_spec),
                                                replica_id=FLAGS.task_index,
                                                total_num_replicas=len(
                                                    worker_spec),
                                                use_locking=True)
        train_op = rep_op.apply_gradients(gradient,
                                          global_step=global_step)
        init_token_op = rep_op.get_init_tokens_op()
        chief_queue_runner = rep_op.get_chief_queue_runner()
    else:
        # 异步模式计算更新梯度 给变量运用梯度
        """这是mini()方法的第二部分，它返回一个运用梯度变化的操作
        def apply_gradients(self, grads_and_vars, global_step=None, name=None)
        grads_and_vars：由`compute_gradients()`返回的梯度或变量的列表对
        global_step：（可选）变量被更新之后加一
        name：返回操作的名称
        """
        train_op = optimizer.apply_gradients(gradient,
                                             global_step=global_step)
    #预测值
    predict = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values)
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), model.targets))
    var_op = tf.global_variables()
    var_trainable_op = tf.trainable_variables()
    saver=tf.train.Saver()
    #判断是否是主节点
    if FLAGS.sync_replicas:
        local_init_op = optimizer.local_step_init_op
        if is_chief:
            #所有的进行计算的工作节点里的一个主工作节点（chief）
            #这个主节点负责初始化参数，模型的保存，概要的保存等
            local_init_op=optimizer.chief_init_op
        ready_for_local_init_op=optimizer.ready_for_local_init_op

        #同步训练模式所需的初始令牌和主队列
        chief_queue_runner=optimizer.get_chief_queue_runner()
        sync_init_op=optimizer.get_init_tokens_op()

    #创建生成日志的目录
    train_dir = tempfile.mkdtemp()
    # 初始化操作
    init_op=tf.global_variables_initializer()
    if FLAGS.sync_replicas:
        """
        创建一个监管程序，用于统计训练模型过程中的信息
        logdir是保存和加载模型的路径
        启动就会去这个logdir目录看是否有检查点文件，有的话自动加载
        没有就用init_op指定的初始化参数
        主工作节点（chief）负责模型初始化等工作
        在这个工程中，其他工作节点等待主节点完成初始化等工作，初始化完成后，一起开始训练数据
        global_step的值所有计算节点共享的
        在执行损失函数最小值时自动加1，通过gloab_step能知道所有节点一共计算啦多少步
        """
        sv=tf.train.Supervisor(
            is_chief=is_chief,
            logdir=train_dir,
            init_op=init_op,
            local_init_op=local_init_op,
            ready_for_local_init_op=ready_for_local_init_op,
            global_step=global_step
        )
    else:
        sv=tf.train.Supervisor(
            is_chief=is_chief,
            logdir=train_dir,
            init_op=init_op,
            recovery_wait_secs=1,
            global_step=global_step,
            saver=saver
        )

    # 在创建会话时，设置属性allow_soft_placement为True，所有的操作会默认使用其被指定的设备如：GPU
    # 如果该操作函数没有GPU实现时，会自动使用CPU设备
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index]
    )
    # 主工作节点（chief），即task_inde为0的节点将会初始化会话
    # 其余的工作节点会等待会话被初始化后就行计算

    if is_chief:
        print ("Worker %d: Initializing session ..." % FLAGS.task_index)
    else:
        print ("Worker %d: Waiting for session to be initialized ..." % FLAGS.task_index)

    if FLAGS.existing_servers:
        server_grpc_url = "grpc://" + worker_spec[FLAGS.task_index]
        print ("Using existing server at : %s" % server_grpc_url)

        # 创建TensorFlow 会话对象，用于执行TensorFlow图计算
        # prepare_or_waite_for_session需要参数初始化完成且主节点也准备好，才开始训练
        sess = sv.prepare_or_wait_for_session(server_grpc_url, config=sess_config)
    else:
        sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
    print ("Worker %d: Session initialization complete." % FLAGS.task_index)

    """开始执行分布式训练"""
    #time_begin = time.time()
    time_begin = datetime.datetime.now()
    print ("Training begins @ %s" % str(time_begin))
    #sess.run(init_op)
    #saver = tf.train.Saver(var_op)
    train_cost = train_ler = 0
    cur_epoch=0


    while True:
        start = datetime.datetime.now()
        for batch in range(num_batches_per_epoch):
            feed = {model.input_data: train_inputs[batch * batch_size: (batch + 1) * batch_size],
                    model.targets: train_targets[batch],
                    model.seq_len: train_seq_len[batch * batch_size: (batch + 1) * batch_size]}
            batch_cost, _,step,ler_cost = sess.run([cost, train_op,global_step,ler], feed_dict=feed)

            train_cost += batch_cost * batch_size
            train_ler += ler_cost * batch_size

        sum_val_cost = sum_val_ler = 0.

        train_cost /= num_examples
        train_ler /= num_examples
        cur_epoch+=1
        log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.23s}"
        print(log.format(cur_epoch , n_epoch, train_cost, train_ler, sum_val_cost / num_batches_per_epoch,
                         sum_val_ler / num_batches_per_epoch, str(datetime.datetime.now() - start)))
        # save("log"+str(cur_epoch), "第" + str(cur_epoch) + "结束于耗时" + str(datetime.datetime.now() - start) + """
        #                         """)
        # save models
        if cur_epoch % 5 == 0:
            saver.save(sess, os.path.join(os.getcwd(), 'model', 'speech.module'), global_step=cur_epoch)
        if step >= num_batches_per_epoch*n_epoch:
            time_end = datetime.datetime.now()
            print ("all training over at @"+str(time_end)+" last :"+str(time_end-time_begin))
            break

    """*******华丽分割线****************"""




