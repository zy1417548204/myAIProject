# -*- coding:utf-8 -*-
import datetime
import time
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
##定义学习率
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
#训练2轮
tf.app.flags.DEFINE_integer('training_epochs', 2,
                            'Training epochs for every thread')


# Hyper-parameters setting  定义超参数
LEARNING_RATE = FLAGS.learning_rate
TRAINING_EPOCHS = FLAGS.training_epochs

def main():
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    #task_index = FLAGS.task_index
    # Define variable  定义变量
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, name="x")

    with tf.name_scope('weights'):
        target_w = tf.Variable(2.0, name='target_w')
        w_list = [tf.Variable(2.0, name='target_w') for i in range(2)]
        w = w_list[0]

    with tf.name_scope('output'):
        y = tf.multiply(x, w, name='y')

    with tf.name_scope('real_output'):
        y_ = tf.placeholder(tf.float32, name="y_")




    with tf.name_scope('gradient'):
        loss = tf.reduce_mean(tf.square(y_ - y))  # MSE loss 和值求平均（损失函数）


    # specify optimizer  指定优化器
    with tf.name_scope('train'):
        # optimizer is an "operation" which we can execute in a session
        # 优化器是一个我们能够执行会话的操作
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    # create a summary for network gradients  为网络梯度创建一个摘要
    init_op = tf.initialize_all_variables()



    with tf.Session() as sess:

        sess.run(init_op)
        epoch = 0
        start =datetime.datetime.now()
        for epoch in range(100000):
            init_w=sess.run(w)
            start_time_epoch = datetime.datetime.now()
            for i in range(3):
                start_time = datetime.datetime.now()
                # print("task%d - epoch%d: " % (task_index, epoch), '  ')
                x_i = i
                y_real = 10 + i

                _,loss_i = sess.run([optimizer,loss],feed_dict={x: x_i, y_: y_real})

                #loss_i = sess.run(loss, feed_dict={x: x_i, y_: y_real})

                end_time = datetime.datetime.now()
                # print (
                # "start_time: " + str(start_time) + ", x_i: " + str(x)  + ", y_real: " + str(
                #     y_real) + ", loss_i: " + str(loss_i)  + ", end_time:" + str(end_time))


            loss2 = sess.run(loss,feed_dict={x: x_i, y_: y_real} )
            end_time = str(datetime.datetime.now())
            new_w = sess.run(w)

            print("start_time:%s    end_time:%s     gradient:%s   Final States of weight in epoch%d: " % (
            start_time_epoch, end_time,str(new_w-init_w), epoch), new_w)
            epoch+=1

    end = datetime.datetime.now()
    print(end - start)
    pass
if __name__ == "__main__":
    main()