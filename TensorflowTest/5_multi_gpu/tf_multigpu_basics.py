from __future__ import print_function

'''
Basic Multi GPU computation example using TensorFlow library.
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

'''
This tutorial requires your machine to have 2 GPUs
"/cpu:0": The CPU of your machine.
"/gpu:0": The first GPU of your machine
"/gpu:1": The second GPU of your machine
'''

import numpy as np
import tensorflow as tf
import datetime

log_device_placement = True

nb_mulply = 10

# compute A^n + B^n on 2 GPUs
A = np.random.rand(10000, 10000).astype('float32')
B = np.random.rand(10000, 10000).astype('float32')

c1 = []
c2 = []

def matpow(M, n):
    if n < 1:
        return M
    else:
        return tf.matmul(M, matpow(M, n-1))

# single GPU
with tf.device('/gpu:0'):
    a = tf.placeholder(tf.float32, [10000, 10000])
    b = tf.placeholder(tf.float32, [10000, 10000])
    c1.append(matpow(a, nb_mulply))
    c1.append(matpow(b, nb_mulply))

with tf.device('/cpu:0'):
    sum = tf.add_n(c1)

t1_1 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
    sess.run(sum, feed_dict={a: A, b: B})
t2_1 = datetime.datetime.now()

# multiple GPUs
with tf.device('/gpu:0'):
    a = tf.placeholder(tf.float32, [10000, 10000])
    c2.append(matpow(a, nb_mulply))
with tf.device('/gpu:1'):
    b = tf.placeholder(tf.float32, [10000, 10000])
    c2.append(matpow(b, nb_mulply))
with tf.device('/cpu:0'):
    sum = tf.add_n(c2)

t1_2 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
    sess.run(sum, feed_dict={a: A, b: B})
t2_2 = datetime.datetime.now()

print('Single GPU time:' + str(t2_1-t1_1))
print('Double GPU time:' + str(t2_2-t1_2))
# .. Result on cores with 2 GTX-1080
#   Single GPU time:0:00:05.925822
#   Double GPU time:0:00:03.218107