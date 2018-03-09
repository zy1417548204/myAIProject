
# -*- coding:utf-8 -*-
# Created by Enigma on 2016/9/26
"""
Verify the mechanism of gradients update operation during asynchronous training in between-graph approach.
证明在异步训练图内模式中梯度更新操作机制
Run:
# ps
/opt/anaconda3/bin/python async_grad_test.py --ps_hosts=localhost:1024 --worker_hosts=localhost:1025,localhost:1026 --job_name=ps --task_index=0
# workers
/opt/anaconda3/bin/python async_grad_test.py --ps_hosts=localhost:1024 --worker_hosts=localhost:1025,localhost:1026 --job_name=worker --task_index=0
/opt/anaconda3/bin/python async_grad_test.py --ps_hosts=localhost:1024 --worker_hosts=localhost:1025,localhost:1026 --job_name=worker --task_index=1
"""
import datetime
import time
import tensorflow as tf

# Define hyper-parameters 定义超参数

FLAGS = tf.app.flags.FLAGS
##定义学习率
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
#训练2轮
tf.app.flags.DEFINE_integer('training_epochs', 100000,
                            'Training epochs for every thread')
#3个线程
tf.app.flags.DEFINE_integer('thread_steps', 3, 'Steps run before sync gradients.')

# Define missions parameters 定义任务参数
#参数服务器，用一号机器作为参数服务器 47.95.32.212
tf.app.flags.DEFINE_string("ps_hosts", "47.95.32.212:2223",
                           "Comma-separated list of hostname:port pairs")
#工作节点服务器 2，3号机器
tf.app.flags.DEFINE_string("worker_hosts", "47.95.32.225:2223,47.95.33.5:2223",
                           "Comma-separated list of hostname:port pairs")
#作业名称
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
#任务名称
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
#检查点目录
tf.app.flags.DEFINE_string("logs_path", "checkpoint/async_grads",
                           "Path to store performance_log")


# Hyper-parameters setting  定义超参数
LEARNING_RATE = FLAGS.learning_rate
TRAINING_EPOCHS = FLAGS.training_epochs
THREAD_STEPS = FLAGS.thread_steps
LOGS_PATH = FLAGS.logs_path
WORKER_NUM = 2


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    #创建集群服务器
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Allow GPU memory grow  允许GPU内存增加 打印相应日志
    server_config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True),
        log_device_placement=True)
    server = tf.train.Server(cluster, config=server_config,
                             job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        with tf.device('/job:ps/task:%d' % FLAGS.task_index):
            server.join()

    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)) as device:

            global_step = tf.get_variable('global_step', [],
                                          initializer=tf.constant_initializer(0),
                                          trainable=False)
            task_index = FLAGS.task_index
            # Define variable  定义变量
            with tf.name_scope('input'):
                x = tf.placeholder(tf.float32, name="x")

            with tf.name_scope('weights'):
                target_w = tf.Variable(2.0, name='target_w')
                w_list = [tf.Variable(2.0, name='target_w') for i in range(WORKER_NUM)]
                w = w_list[task_index]

            with tf.name_scope('output'):
                y = tf.multiply(x, w, name='y')

            with tf.name_scope('real_output'):
                y_ = tf.placeholder(tf.float32, name="y_")



            with tf.name_scope('gradient'):
                loss = tf.reduce_mean(tf.square(y_ - y))  # MSE loss 和值求平均（损失函数）
                # # gradient_all = optimizer.compute_gradients(loss)  # gradient of network (with NoneType)
                # #计算梯度
                # gradient_all = optimizer.compute_gradients(loss)  # gradient of network (with NoneType)
                # #去除那些g为空的值
                # grads_vars = [v for (g, v) in gradient_all if g is not None]  # all variable that has gradients
                # #重新计算梯度
                # gradient = optimizer.compute_gradients(loss, grads_vars)  # gradient of network (without NoneType)
                #
                # grads_holder = [(tf.placeholder(tf.float32, shape=g.get_shape()), v)
                #                 for (g, v) in gradient]
                # #更新梯度
                # train_op = optimizer.apply_gradients(grads_holder, global_step=global_step)

            # specify optimizer  指定优化器
            with tf.name_scope('train'):
                # optimizer is an "operation" which we can execute in a session
                # 优化器是一个我们能够执行会话的操作
                optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss,global_step=global_step)
            # create a summary for network gradients  为网络梯度创建一个摘要
            init_op = tf.initialize_all_variables()
            #epoch_init_op = w.assign(target_w)
            #w_addup = tf.placeholder(tf.float32)
            #epoch_update_op = target_w.assign_add(w_addup)

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir=LOGS_PATH,
                                 init_op=init_op,
                                 global_step=global_step,
                                 save_model_secs=180)

        with sv.prepare_or_wait_for_session(server.target) as sess:
            # create performance_log writer object (this will performance_log on every machine)
            # perform training cycles
            # time.sleep(sleep_time)

            local_step = 0
            start = datetime.datetime.now()
            while True:
                #_ = sess.run(epoch_init_op)
                init_w=w.eval()
                start_time_epoch = datetime.datetime.now()

                for i in range(THREAD_STEPS):
                    start_time = datetime.datetime.now()
                    #print("task%d - epoch%d: " % (task_index, epoch), '  ')
                    x_i = i
                    y_real = 10 + i
                    #print('x_i: ', x_i, '. ')

                    #预测值
                    _, loss_i = sess.run([optimizer, loss], feed_dict={x: x_i, y_: y_real})
                    #print('y_i: ', y_i, '. ')
                    #loss_i = sess.run(loss, feed_dict={x: x_i, y_: y_real})

                    end_time = datetime.datetime.now()
                    # print (
                    #     "start_time: " + str(start_time) + ", x_i: " + str(x) + ", y_real: " + str(
                    #         y_real) + ", loss_i: " + str(loss_i) + ", end_time:" + str(end_time))
                    #grads.append(grad_i)
                    #time.sleep(0.5)

                    # print("States of w in task%d - thread_step%d: " % (FLAGS.task_index, i), w.eval())
                    # time.sleep(2)

                # calculate total gradients
                grads_sum = {}
                # add up dθ

                # for i in range(len(grads_holder)):
                #     k = grads_holder[i][0]
                #     if k is not None:
                #         grads_sum[k] = sum([g[i][0] for g in grads])
                start_time = str(datetime.datetime.now())
                loss2,step = sess.run([loss,global_step], feed_dict={x: x_i, y_: y_real})
                end_time = str(datetime.datetime.now())
                new_w=w.eval()
                # print("start_time:%s    end_time:%s   grident:%s  Final States of weight in epoch%d: " % (
                #     start_time_epoch, end_time, str(new_w-init_w),step), new_w)
                local_step+=1

                if step>=FLAGS.training_epochs:
                    break
        end = datetime.datetime.now()
        print(end - start)
        sv.stop()
        print("done")


if __name__ == "__main__":
    tf.app.run()