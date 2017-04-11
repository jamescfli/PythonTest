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
from ptb_iterator import ptb_iterator


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


def gen_epochs(n, num_steps, batch_size):
    for i in range(n):
        # yield reader.ptb_iterator(data, batch_size, num_steps)
        yield ptb_iterator(data, batch_size, num_steps)
        # yield ptb_producer(data, batch_size, num_steps)


def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()    # just in case
    tf.reset_default_graph()


def train_network(g, num_epochs, num_steps=200, batch_size=32, verbose=True, save=False):
    tf.set_random_seed(2345)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps, batch_size)):
            training_loss = 0
            steps = 0
            training_state = None
            for X, Y in epoch:
                steps += 1
                feed_dict = {g['x']: X, g['y']: Y}
                # print('X.shape:', X.shape, 'Y.shape:', Y.shape)
                if training_state is not None:
                    feed_dict[g['init_state']] = training_state
                training_loss_, training_state, _ = sess.run([g['total_loss'],
                                                              g['final_state'],
                                                              g['train_step']], feed_dict)
                training_loss += training_loss_
            if verbose:
                print("Average training loss for Epoch", idx, ":", training_loss/steps)
            training_losses.append(training_loss/steps)
        if isinstance(save, str):
            g['saver'].save(sess, save)
    return training_losses

# output ('Data length:', 1115394)


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

# t = time.time()
# build_basic_rnn_graph_with_list()
# print('It took', time.time()-t, 'secs to build the basic rnn graph')   # 15.033665895462036 secs on MBP


def build_multilayer_lstm_graph_with_list(
        state_size = 100,
        num_classes = vocab_size,
        batch_size = 32,
        num_steps = 200,
        num_layers = 3,
        learning_rate = 1e-4):
    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])
    rnn_inputs = [tf.squeeze(i) for i in tf.split(tf.nn.embedding_lookup(embeddings, x), num_steps, 1)]

    cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    # .. state_is_tuple: True by default
    # .. If True, accepted and returned states are 2-tuples of the c_state and m_state.
    # .. If False, they are concatenated along the column axis.
    # .. The latter behavior will soon be deprecated.
    cell = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]

    y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(y, num_steps, 1)]

    loss_weights = [tf.ones([batch_size]) for i in range(num_steps)]
    losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(logits, y_as_list, loss_weights)
    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(x = x,
                y = y,
                init_state = init_state,
                final_state = final_state,
                total_loss = total_loss,
                train_step = train_step)

# t = time.time()
# build_multilayer_lstm_graph_with_list()
# print('It took', time.time()-t, 'secs to build the multilayer graph.')      # 77.6839771270752 secs


def build_multilayer_lstm_graph_with_dynamic_rnn(
        state_size = 100,
        num_classes = vocab_size,
        batch_size = 32,
        num_steps = 200,
        num_layers = 3,
        learning_rate = 1e-4):
    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])

    # note that inputs are no longer a list, but a tensor with batch_size + num_steps + state_size
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    cell = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    # reshape rnn_outputs and y so we can get the logits in a single matmul
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(y, [-1])

    logits = tf.matmul(rnn_outputs, W) + b

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(x = x,
                y = y,
                init_state = init_state,
                final_state = final_state,
                total_loss = total_loss,
                train_step = train_step)

# t = time.time()
# build_multilayer_lstm_graph_with_dynamic_rnn()
# print('It took', time.time() - t, 'secs to build the graph.')   # 2.016392946243286 secs

# # dynamic_rnn speeds things up
# g = build_multilayer_lstm_graph_with_dynamic_rnn()
# t = time.time()
# train_network(g, 3)
# print('It took', time.time()-t, 'secs to train 3 epochs.')  # 314.98641204833984 secs

# # compare with static_rnn
# g = build_multilayer_lstm_graph_with_list()
# t = time.time()
# train_network(g, 3)
# print('It took', time.time()-t, 'secs to train 3 epochs.')  # 252.21862721443176 secs < 314 secs


# use scan with an LSTM so as to compare to the dynamic_rnn using Tensorflow above
def build_multilayer_lstm_graph_with_scan(
        state_size = 100,
        num_classes = vocab_size,
        batch_size = 32,
        num_steps = 200,
        num_layers = 3,
        learning_rate = 1e-4):
    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])

    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    cell = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    init_state = cell.zero_state(batch_size, tf.float32)
    # scan produces rnn_outputs with shape [num_steps, batch_size, state_size] from [batch_size, num_steps, state_size]
    rnn_outputs, final_states = tf.scan(lambda a, x: cell(x, a[1]),
                                       tf.transpose(rnn_inputs, [1,0,2]),   # see above explanation
                                       initializer=(tf.zeros([batch_size, state_size]), init_state))

    # there may be a better way to do this
    final_state = tuple([tf.contrib.rnn.core_rnn_cell.LSTMStateTuple(
        tf.squeeze(tf.slice(c, [num_steps-1, 0, 0], [1, batch_size, state_size])),
        tf.squeeze(tf.slice(h, [num_steps-1, 0, 0], [1, batch_size, state_size])))
        for c, h in final_states])

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(tf.transpose(y, [1,0]), [-1])

    logits = tf.matmul(rnn_outputs, W) + b

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(x = x,
                y = y,
                init_state = init_state,
                final_state = final_state,
                total_loss = total_loss,
                train_step = train_step)

# t = time.time()
# g = build_multilayer_lstm_graph_with_scan()
# print("It took", time.time() - t, "secs to build the graph.")   # 2.1207070350646973 secs
# t = time.time()
# train_network(g, 3)
# print("It took", time.time() - t, "secs to train for 3 epochs.")  # 355.96185183525085 secs > dynamic rnn
# # .. scan was only marginally slower than using dynamic_rnn, and gives us the flexibility e.g. create a skip connection

# cell = tf.contrib.rnn.core_rnn_cell.BasicRNNCell(state_size)
# # to
# cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(state_size)
# # or
# cell = tf.contrib.rnn.core_rnn_cell.GRUCell(state_size)
