# biggest change is that many of the RNN cells and functions are in the tf.contrib.rnn module
# also a change to the ptb_iterator @ https://gist.github.com/spitis/2dd1720850154b25d2cec58d4b75c4a0

# build a character-level language model to generate character sequences
# a good idea to restrict the vocabulary

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
# from tensorflow.models.rnn.ptb import reader
from ptb_iterator import ptb_iterator, shuffled_ptb_iterator

# file_url = 'https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt'
file_name = 'tiny-shakespeare.txt'

with open(file_name, 'r') as f:
    raw_data = f.read()
    print("Data length:", len(raw_data))

vocab = set(raw_data)
vocab_size = len(vocab)     # all words
idx_to_vocab = dict(enumerate(vocab))
vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))

data = [vocab_to_idx[c] for c in raw_data]
del raw_data


# def gen_epochs(n, num_steps, batch_size):
#     for i in range(n):
#         yield ptb_iterator(data, batch_size, num_steps)


def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()    # just in case
    tf.reset_default_graph()


# def train_network(g, num_epochs, num_steps = 200, batch_size = 32, verbose = True, save = False):
#     tf.set_random_seed(2345)
#     with tf.Session() as sess:
#         sess.run(tf.initialize_all_variables())
#         training_losses = []
#         for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps, batch_size)):
#             training_loss = 0
#             steps = 0
#             training_state = None
#             for X, Y in epoch:
#                 steps += 1
#                 feed_dict = {g['x']: X, g['y']: Y}
#                 if training_state is not None:
#                     feed_dict[g['init_state']] = training_state
#                 training_loss_, training_state, _ = sess.run([g['total_loss'],
#                                                               g['final_state'],
#                                                               g['train_step']], feed_dict)
#                 training_loss += training_loss_
#             if verbose:
#                 print("Average training loss for Epoch", idx, ":", training_loss/steps)
#             training_losses.append(training_loss/steps)
#         if isinstance(save, str):
#             g['saver'].save(sess, save)
#     return training_losses
#
# # output ('Data length:', 1115394)


# tf.scan and dynamic_rnn to speed up
# to capture much longer dependencies we build a graph that is 200 time steps wide
def build_basic_rnn_graph_with_list(
        state_size = 100,
        num_classes = vocab_size,
        batch_size = 32,
        num_steps = 200,
        learning_rate = 1e-4):
    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    x_one_hot = tf.one_hot(x, num_classes)
    rnn_inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(x_one_hot, num_steps, 1)]

    # cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
    cell = tf.contrib.rnn.core_rnn_cell.BasicRNNCell(state_size)
    init_state = cell.zero_state(batch_size, tf.float32)
    # rnn_outputs, final_state = tf.nn.rnn(cell, rnn_inputs, initial_state=init_state)
    rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=init_state)


    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = [tf.matmul(rnn_output, W)+b for rnn_output in rnn_outputs]

    y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(y, num_steps, 1)]

    loss_weights = [tf.ones([batch_size]) for i in range(num_steps)]    # equal weights
    # losses = tf.nn.seq2seq.sequence_loss_by_example(logits, y_as_list, loss_weights)
    losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(logits=logits, targets=y_as_list, weights=loss_weights)
    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(x = x,
                y = y,
                init_state = init_state,
                final_state = final_state,
                total_loss = total_loss,
                train_step = train_step)

t = time.time()
build_basic_rnn_graph_with_list()
print('It took', time.time()-t, 'secs to build the graph')   # 15.033665895462036 secs on MBP



