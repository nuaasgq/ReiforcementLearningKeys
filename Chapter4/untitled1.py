# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:10:17 2020

@author: SGQ
"""
import tensorflow as tf
tf.reset_default_graph()

with tf.name_scope('123'):
    with tf.name_scope('456'):
        with tf.variable_scope('789'):
            a = tf.Variable(1, name = 'a')
            print(a.name)
        with tf.variable_scope('789'):
#            b = tf.get_variable('b', 1)
            b = tf.Variable(1, name = 'b')
            print(b.name)
