# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 10:25:03 2020

@author: SGQ
"""
import numpy as np
import tensorflow as tf

tf.reset_default_graph()

data_x = np.array([[0,0], [0,1], [1,0], [1,1]], dtype = np.float32)
data_y = np.array([[0], [1], [1], [0]], dtype = np.float32)

x = tf.placeholder(tf.float32, shape = [None, 2], name = 'imput')

w1 = tf.Variable(tf.random_uniform([2,4], -1, 1), name = 'w1')
b1 = tf.Variable(tf.zeros(4, dtype = np.float32), name = 'b1')
w2 = tf.get_variable('w2', shape = [4,1], initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.zeros(4, dtype = np.float32), name = 'b2')

z1 = tf.sigmoid(tf.matmul(x, w1) + b1)
z2 = tf.sigmoid(tf.matmul(z1, w2) + b2)

y = tf.placeholder(tf.float32, shape = [None, 1], name = 'output')

loss = tf.nn.l2_loss(z2 - y)

opt = tf.train.GradientDescentOptimizer(0.05)
#该函数没有具体的返回值，但是可以用于开启会话，更新神经网络的参数，实现训练效果
train_op = opt.minimize(loss)#An Operation that updates the variables in var_list. If global_step was not None, that operation also increments global_step.


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iter in range (100000):
        loss_val, _ = sess.run([loss, train_op], feed_dict = {x:data_x, y:data_y})
        if iter % 10000 == 0:
            print('{}:loss={}'.format(iter, loss_val))
            
#graph = tf.get_default_graph()
#print(graph.get_operations()) 