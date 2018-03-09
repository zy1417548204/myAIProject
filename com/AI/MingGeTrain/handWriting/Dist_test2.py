# coding=utf-8
import tempfile

import numpy as np
import tensorflow as tf
import os
import math
import time
from tensorflow.examples.tutorials.mnist import input_data

"""
依照书上手写，手写数字识别（/tmp/mnist-data）运用回归模型
"""
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
flags.DEFINE_integer("hidden_units",100,"Number of units in the hidden layer of the NN")
#训练次数
flags.DEFINE_integer("train_steps",200,"Number of (gloabal) training steps to perform")
#每一批次样本数
flags.DEFINE_integer("batch_size",100,"Training batch size")
#学习率
flags.DEFINE_float("learning_rate",0.01,"Learning rate")
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
flags.DEFINE_string("worker_hosts","47.95.32.225:2222,47.95.33.5:2222","Comma_separated list of hostname :port pairs")
#本作业是工作节点还是参数服务器
flags.DEFINE_string("job_name",None,"job name:worker or ps")
FLAGS=flags.FLAGS
IMAGE_PIXELS=28

#读取集群描述信息
ps_spec=FLAGS.ps_hosts.split(",")
worker_spec=FLAGS.worker_hosts.split(",")
"""临时添加"""
num_workers=len(worker_spec)+1
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
    cpu = 0
    worker_device ="/job:worker/task:%d/gpu:%d"%(FLAGS.task_index,cpu)
#在这个with 语句之下定义的参数，会自动分配到参数服务器上去定义，如果有多个参数服务器就轮流分配
with tf.device(
    tf.train.replica_device_setter(
        worker_device=worker_device,
        ps_device="/job:ps/cpu:0",
        cluster=cluster
    )
):
    #定义全局步长，默认值为0
    global_step = tf.Variable(0,name="global_step",trainable=False)
    #定义隐藏层参数变量，这里是全链接神经网络隐藏层
    hid_w=tf.Variable(
        tf.truncated_normal(
            [IMAGE_PIXELS*IMAGE_PIXELS,FLAGS.hidden_units],
            stddev=1.0/IMAGE_PIXELS
        ),name="hid_w"
    )
    hid_b=tf.Variable(tf.zeros([FLAGS.hidden_units]),name="hid_b")
    #定义SoftMax回归层参数变量
    sm_w=tf.Variable(
        tf.truncated_normal(
            [FLAGS.hidden_units,10],
            stddev=1.0 / math.sqrt(FLAGS.hidden_units)#math.sqrt(FLAGS.hidden_units)
        ),name="sm_w"
    )
    sm_b=tf.Variable(tf.zeros([10]),name="sm_b")
    #定义模型输入数据变量
    x=tf.placeholder(tf.float32,[None,IMAGE_PIXELS*IMAGE_PIXELS])
    y_=tf.placeholder(tf.float32,[None,10])
    #构建隐藏层
    hid_lin=tf.nn.xw_plus_b(x,hid_w,hid_b)
    hid=tf.nn.relu(hid_lin)
    #构建损失函数和优化器
    y=tf.nn.softmax(tf.nn.xw_plus_b(hid,sm_w,sm_b))
    cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
    #异步训练模式：自己计算完梯度就去更新参数，不同副本之间不会去协调进度
    opt=tf.train.AdamOptimizer(FLAGS.learning_rate)

    #同步训练模式
    if FLAGS.sync_replicas:
        if FLAGS.replicas_to_aggregate is None:
            replicas_to_aggregate=num_workers
        else:
            replicas_to_aggregate =FLAGS.replicas_to_aggregate
        #使用SyncReplicasOptimizer作为优化器，并且在图间复制的情况下
        #在图内复制的情况下将所有的梯度平均就可以啦
        opt=tf.train.SyncReplicasOptimizer(
            opt,
            replicas_to_aggregate=replicas_to_aggregate,
            total_num_replicas=num_workers,
            name="mnist_sync_replicas"
        )
    train_step=opt.minimize(cross_entropy,global_step=global_step)



    if FLAGS.sync_replicas:
        local_init_op = opt.local_step_init_op
        if is_chief:
            #所有的进行计算的工作节点里的一个主工作节点（chief）
            #这个主节点负责初始化参数，模型的保存，概要的保存等
            local_init_op=opt.chief_init_op
        ready_for_local_init_op=opt.ready_for_local_init_op

        #同步训练模式所需的初始令牌和主队列
        chief_queue_runner=opt.get_chief_queue_runner()
        sync_init_op=opt.get_init_tokens_op()
    """临时添加"""
    #tempfile=FLAGS.data_dir

    init_op=tf.global_variables_initializer()

    # 评估训练好的模型
    # 计算预测值和真实值
    correct_predition = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # 布尔型转化为浮点数并取平均值，得到准确率
    accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))

    train_dir=tempfile.mkdtemp()
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
            global_step=global_step
        )
    #在创建会话时，设置属性allow_soft_placement为True，所有的操作会默认使用其被指定的设备如：GPU
    #如果该操作函数没有GPU实现时，会自动使用CPU设备
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_filters=["/job:ps","/job:worker/task:%d"%FLAGS.task_index]
    )
    #主工作节点（chief），即task_inde为0的节点将会初始化会话
    #其余的工作节点会等待会话被初始化后就行计算

    if is_chief:
        print ("Worker %d: Initializing session ..." % FLAGS.task_index)
    else:
        print ("Worker %d: Waiting for session to be initialized ..." % FLAGS.task_index)

    if FLAGS.existing_servers:
        server_grpc_url="grpc://"+worker_spec[FLAGS.task_index]
        print ("Using existing server at : %s" % server_grpc_url)

        #创建TensorFlow 会话对象，用于执行TensorFlow图计算
        #prepare_or_waite_for_session需要参数初始化完成且主节点也准备好，才开始训练
        sess = sv.prepare_or_wait_for_session(server_grpc_url,config=sess_config)
    else:
        sess = sv.prepare_or_wait_for_session(server.target,config=sess_config)
    print ("Worker %d: Session initialization complete." % FLAGS.task_index)

    if FLAGS.sync_replicas and is_chief:
        sess.run(sync_init_op)
        sv.start_queue_runners(sess,[chief_queue_runner])

    #执行分布式训练模型
    time_begin=time.time()
    print ("Training begins @ %f" % time_begin)

    local_step=0
    """临时加上    one_hot：标记是指一个长度为n的数组，只有一个元素是1.0,其他元素是0.0"""
    mnist = input_data.read_data_sets("/tmp/mnist-data/",one_hot=True)
    while True:
        #读入MNIST的训练数据集，默认每次为100张图片
        batch_xs,batch_ys=mnist.train.next_batch(FLAGS.batch_size)
        train_feed ={x:batch_xs,y_:batch_ys}

        _,step = sess.run([train_step,global_step],feed_dict=train_feed)
        local_step+=1

        now = time.time()
        print ("%f: Worker %d: training step %d done (global step:%d)" % (now,FLAGS.task_index,local_step,step))



        if step >= FLAGS.train_steps:
            break

    time_end = time.time()
    print ("Training ends @ %f " % time_end)

    train_time = time_end-time_begin
    print ("Training time: @ %f s" % train_time)

    #读入MNIST的验证数据，计算验证的交叉熵
    val_feed ={x: mnist.validation.images,y_: mnist.validation.labels}
    val_xent =sess.run(cross_entropy,feed_dict=val_feed)
    print ("After %d training step(s), validation cross entropy = %g" % (FLAGS.train_steps,val_xent))

    # 评估训练好的模型
    # 计算预测值和真实值
    #correct_predition = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # 布尔型转化为浮点数并取平均值，得到准确率
    #accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))
    # 计算模型在测试集上的准确率
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
