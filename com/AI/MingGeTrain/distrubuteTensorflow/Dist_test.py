# coding=utf-8
import numpy as np
import tensorflow as tf
import datetime
import os
#忽视忽视警告，并屏蔽警告
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Define parameters
FLAGS = tf.app.flags.FLAGS
#定义学习率
tf.app.flags.DEFINE_float('learning_rate', 0.00003, 'Initial learning rate.')
#每多少步有效并且打印损失
tf.app.flags.DEFINE_integer('steps_to_validate', 1000,
                            'Steps to validate and print loss')

# For distributed
#定义参数服务器线上001
tf.app.flags.DEFINE_string("ps_hosts", "47.95.32.212:2222",
                           "Comma-separated list of hostname:port pairs")
#定义worker节点 002+003+004
tf.app.flags.DEFINE_string("worker_hosts", "47.95.32.225:2222,47.95.33.5:2222,47.94.41.45:2222",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
#task_index
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
#判断同步模式，异步模式
tf.app.flags.DEFINE_integer("issync", 0, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")

# Hyperparameters
learning_rate = FLAGS.learning_rate
steps_to_validate = FLAGS.steps_to_validate


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    #tf.train.Server（self,
               #server_or_cluster_def,    创建的集群
               #job_name=None,            作业名称（可选）
               #task_index=None,          任务名称（可选）
               #protocol=None,            协议（可选，默认grpc）
               #config=None,              配置（可选）指定默认值的`tf.ConfigProto`在此服务器上运行的所有会话的配置选项。
               #start=True）              创建之后默认开启服务（可选）
    """sess = tf.Session(config=tf.ConfigProto(
    allow_soft_placement=True, log_device_ placement=True))"""
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index,config=tf.ConfigProto(log_device_placement=True))

    issync = FLAGS.issync
    if FLAGS.job_name == "ps":
        server.join()

    elif FLAGS.job_name == "worker":
        #tf.device(device_name_or_function）  `Graph.device()`的包装，传入设备名称或函数
        # ，返回get_default_graph().device(device_name_or_function)
        """def replica_device_setter(ps_tasks=0, ps_device="/job:ps",
                          worker_device="/job:worker", merge_devices=True,
                          cluster=None, ps_ops=None, ps_strategy=None)
                          返回一个”设备函数“，在构建副本图时使用。
        """
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
            #全局步数
            global_step = tf.Variable(0, name='global_step', trainable=False)

            input = tf.placeholder("float")
            label = tf.placeholder("float")

            weight = tf.get_variable("weight", [1], tf.float32, initializer=tf.random_normal_initializer())
            biase = tf.get_variable("biase", [1], tf.float32, initializer=tf.random_normal_initializer())
            pred = tf.multiply(input, weight) + biase

            loss_value = loss(label, pred)
            #梯度下降优化器
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            #为变量列表里的变量计算梯度的损失
            """这是minim()的第一部分，它返回一个列表对（梯度，变量）
            其中梯度是可变的的是一个张量，如果没有梯度的话，给定变量"""
            grads_and_vars = optimizer.compute_gradients(loss_value)
            if issync == 1:
                # 同步模式计算更新梯度
                rep_op = tf.train.SyncReplicasOptimizer(optimizer,
                                                        replicas_to_aggregate=len(
                                                            worker_hosts),
                                                        replica_id=FLAGS.task_index,
                                                        total_num_replicas=len(
                                                            worker_hosts),
                                                        use_locking=True)
                train_op = rep_op.apply_gradients(grads_and_vars,
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
                train_op = optimizer.apply_gradients(grads_and_vars,
                                                     global_step=global_step)

            init_op = tf.initialize_all_variables()

            saver = tf.train.Saver()
            #输出包含单个标量值的“Summary”协议缓冲区
            """summary_writer = tf.summary.FileWriter('./mnist_logs')

                summary,codes_batch = sess.run([merged,vgg.relu6], feed_dict=feed_dict)
                summary_writer.add_summary(summary, ii)"""
            summary_writer = tf.summary.FileWriter('./mnist_logs')
            tf.summary.scalar('cost', loss_value)
            summary_op = tf.summary.merge_all()
        #添加检查点路径
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir="./checkpoint/",
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=60,
                                 summary_writer=summary_writer)
        #确保模型已经准备好并且能被使用
        """def prepare_or_wait_for_session(self, master="", config=None,
                                  wait_for_checkpoint=False,
                                  max_wait_secs=7200,
                                  start_standard_services=True)
            在主节点上创建会话，恢复或初始化模型需要，或者等待一个会话准备就绪
            若果担任主节点，同时调用会话管理器启动标准服务
        """
        with sv.prepare_or_wait_for_session(server.target) as sess:
            # 如果是同步模式
            if FLAGS.task_index == 0 and issync == 1:
                sv.start_queue_runners(sess, [chief_queue_runner])
                sess.run(init_token_op)
            step = 0

            while step < 1000000:
                start_time = datetime.datetime.now()
                train_x = np.random.randn(1)
                train_y = 2 * train_x + np.random.randn(1) * 0.33 + 10
                _, loss_v, step = sess.run([train_op, loss_value, global_step],
                                           feed_dict={input: train_x, label: train_y})
                end_time = datetime.datetime.now()
                """添加摘要"""
                #summary_writer.add_summary(loss_v,global_step)
                print ("step: %d, loss: %f,start_time:%s,end_time:%s" % (step, loss_v,start_time,end_time))
                #if step % steps_to_validate == 0:
                    #w, b = sess.run([weight, biase])
                    #duration = time.time() - start_time
                    #print("step: %d, weight: %f, biase: %f, loss: %f" % (step, w, b, loss_v))

        sv.stop()


def loss(label, pred):
    return tf.square(label - pred)


if __name__ == "__main__":
    tf.app.run()
