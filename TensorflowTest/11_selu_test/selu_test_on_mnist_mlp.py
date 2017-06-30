from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler    # transform to 0 mean and unit variance

import numbers
from tensorflow.contrib import layers   # to build layers, regularizers, summaries etc.
from tensorflow.python.framework import ops     # tensor related
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils

import sys
sys.path.append('.')
from selu import selu, dropout_selu

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

# parameters
learning_rate = 0.05
training_epochs = 15
batch_size = 100
display_step = 1

n_hidden_1 = 784
n_hidden_2 = 784
n_input = 784
n_classes = 10

# graph input
x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_classes])
dropout_rate_t = tf.placeholder(tf.float32)
is_training_t = tf.placeholder(tf.bool)

# scale the input
scaler = StandardScaler().fit(mnist.train.images)

# tensorboard
logs_path = '~/tmp'

# model
def mlp_w_selu(x, weights, biases, dropout_rate, is_training):
    layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer1 = selu(layer1)
    layer1 = dropout_selu(layer1, dropout_rate, training=is_training)

    layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
    layer2 = selu(layer2)
    layer2 = dropout_selu(layer2, dropout_rate, training=is_training)

    out_layer = tf.matmul(layer2, weights['out']) + biases['out']   # with linear activation
    return out_layer

def mlp_w_bn_relu(x, weights, biases, dropout_rate, is_training):
    layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer1 = layers.batch_norm(layer1, center=True, scale=True, is_training=is_training_t)
    layer1 = tf.nn.relu(layer1)

    layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
    layer2 = layers.batch_norm(layer2, center=True, scale=True, is_training=is_training_t)
    layer2 = tf.nn.relu(layer2)

    out_layer = tf.matmul(layer2, weights['out']) + biases['out']  # with linear activation
    return out_layer

# initialization with STDDEV of sqrt(1/n)
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=np.sqrt(1/n_input))),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=np.sqrt(1/n_hidden_1))),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=np.sqrt(1/n_hidden_2)))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], stddev=0.0)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], stddev=0.0)),
    'out': tf.Variable(tf.random_normal([n_classes], stddev=0.0))
}

# pred = mlp_w_selu(x, weights, biases, dropout_rate=dropout_rate_t, is_training=is_training_t)
pred = mlp_w_bn_relu(x, weights, biases, dropout_rate=dropout_rate_t, is_training=is_training_t)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# test model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, 'float'))

init = tf.global_variables_initializer()

# for tb, create histogram for weights
tf.summary.histogram("weights2", weights['h2'])
tf.summary.histogram("weights1", weights['h1'])
# cost
tf.summary.scalar('loss', cost)
# accuracy
tf.summary.scalar('accuracy', accuracy)
# merge
merged_summary_op = tf.summary.merge_all()

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)

    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = scaler.transform(batch_x)     # total mean var already calculated
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y,
                                                          dropout_rate_t: 0.0,
                                                          is_training_t: True})
            avg_cost += c/total_batch
        # display
        if epoch % display_step == 0:
            print('Epoch:', '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(avg_cost))
            acc_train, cost_train, summary = sess.run([accuracy, cost, merged_summary_op],
                                                      feed_dict={x: batch_x,
                                                                 y:batch_y,
                                                                 dropout_rate_t: 0.0,
                                                                 is_training_t: False})
            summary_writer.add_summary(summary, epoch)
            print('Train Acc:', acc_train, 'Train Loss:', cost_train)

            batch_x_test, batch_y_test = mnist.test.next_batch(512)
            batch_x_test = scaler.transform(batch_x_test)
            acc_test, cost_test = sess.run([accuracy, cost], feed_dict={x: batch_x_test,
                                                                        y: batch_y_test,
                                                                        dropout_rate_t: 0.0,
                                                                        is_training_t: False})
            print('Test Acc:', acc_test, 'Test Loss:', cost_test, "\n")

# # with selu
# Epoch: 0015 cost= 0.014958596
# Train Acc: 1.0 Train Loss: 0.00565069
# Test Acc: 0.990234 Test Loss: 0.0921781

# # with bn relu
# Epoch: 0015 cost= 0.007487323
# Train Acc: 0.9 Train Loss: 0.291616
# Test Acc: 0.912109 Test Loss: 0.331339