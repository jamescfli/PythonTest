import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
# from tensorflow.models.rnn.ptb import reader
from ptb_iterator import ptb_iterator
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl


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


# layer normalization: function to normalize a 2D tensor along its second dimension
def ln(tensor, scope = None, epsilon = 1e-5):
    assert(len(tensor.get_shape()) == 2)
    m, v = tf.nn.moments(tensor, [1], keep_dims=True)
    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'layer_norm'):
        scale = tf.get_variable('scale',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    LN_initial = (tensor - m) / tf.sqrt(v + epsilon)
    return LN_initial * scale + shift


# apply layer normalization to LSTM
class LayerNormalizedLSTMCell(tf.contrib.rnn.core_rnn_cell.RNNCell):
    def __init__(self, num_units, forget_bias=1.0, activation=tf.nn.tanh):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation

    @property
    def state_size(self):
        return tf.contrib.rnn.core_rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            concat = core_rnn_cell_impl._linear([inputs, h], 4*self._num_units, False)

            i, j, f, o = tf.split(concat, 4, 1)

            # add layer normalization to each gate (before activation)
            i = ln(i, scope='i/')
            j = ln(j, scope='j/')
            f = ln(f, scope='f/')
            o = ln(o, scope='o/')

            new_c = (c*tf.nn.sigmoid(f+self._forget_bias) + tf.nn.sigmoid(i)*self._activation(j))

            # add layer normalization to calculate new hidden state
            new_h = self._activation(ln(new_c, scope='new_h/')) * tf.nn.sigmoid(o)
            new_state = tf.contrib.rnn.core_rnn_cell.LSTMStateTuple(new_c, new_h)

            return new_h, new_state


def build_multilayer_graph_with_lnlstm_cell(
        state_size = 100,
        num_classes = vocab_size,
        batch_size = 32,
        num_steps = 200,
        num_layers = 3,
        learning_rate = 1e-4):
    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])     # to be learnt

    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    cell = LayerNormalizedLSTMCell(state_size)
    cell = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([cell]*num_layers, state_is_tuple=True)

    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    # reshape rnn_outputs and y, for loss calculation
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


g = build_multilayer_graph_with_lnlstm_cell()
t = time.time()
train_network(g, 5)
print("It took", time.time() - t, "secs to train for 5 epochs.")    # 2993 secs, loss 3.316
