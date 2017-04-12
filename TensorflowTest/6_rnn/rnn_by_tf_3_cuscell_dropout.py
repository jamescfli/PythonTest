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


# build a custom RNN cell
class CustomCell(tf.contrib.rnn.core_rnn_cell.RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, num_units, num_weights):
        self._num_units = num_units
        self._num_weights = num_weights

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):   # GRUCell
            with tf.variable_scope("Gates"):
                # start with bias of 1.0 to not reset and not update
                ru = core_rnn_cell_impl._linear(args=[inputs, state],
                                                output_size=2*self._num_units,
                                                bias=True,
                                                bias_start=1.0)
                ru = tf.nn.sigmoid(ru)
                r, u = tf.split(ru, 2, 1)
            with tf.variable_scope("Candidate"):
                lambdas = core_rnn_cell_impl._linear(args=[inputs, state],
                                                     output_size=self._num_weights,
                                                     bias=True)
                lambdas = tf.split(tf.nn.softmax(lambdas), self._num_weights, 1)

                # all of the W_i matrices are created as a single variable
                Ws = tf.get_variable("Ws", shape=[self._num_weights, inputs.get_shape()[1], self._num_units])
                # and then split into multiple tensors
                Ws = [tf.squeeze(i) for i in tf.split(Ws, self._num_weights, 0)]

                candidate_inputs = []

                for idx, W in enumerate(Ws):
                    candidate_inputs.append(tf.matmul(inputs, W) * lambdas[idx])

                Wx = tf.add_n(candidate_inputs)

                # c = tf.nn.tanh(core_rnn_cell_impl._linear([inputs, r*state], self._num_units, True))
                # c = tf.nn.tanh(Wx + core_rnn_cell_impl._linear([r*state], self._num_units, True, scope="second"))
                c = tf.nn.tanh(Wx + core_rnn_cell_impl._linear(args=[r*state],
                                                               output_size=self._num_units,
                                                               bias=True))
            new_h = u * state + (1-u) * c
        return new_h, new_h


# stacks up custom cell to a regular GRU cell (using num_steps = 30,
# since this performs much better than num_steps = 200 after 5 epochs)
def build_multilayer_graph_with_custom_cell(
        cell_type = None,
        num_weights_for_custom_cell = 5,
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

    if cell_type == 'Custom':
        cell = CustomCell(state_size, num_weights_for_custom_cell)
    elif cell_type == "GRU":
        cell = tf.contrib.rnn.core_rnn_cell.GRUCell(state_size)
    elif cell_type == 'LSTM':
        cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    else:
        cell = tf.contrib.rnn.core_rnn_cell.BasicRNNCell(state_size)

    # # add dropout
    # cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(cell, input_keep_prob=global_dropout)

    if cell_type == 'LSTM':
        cell = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([cell]*num_layers, state_is_tuple=True)
    else:
        cell = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([cell]*num_layers)

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

# g = build_multilayer_graph_with_custom_cell(cell_type='GRU', num_steps=30)
# t = time.time()
# train_network(g, 5, num_steps=30)
# print("It took", time.time() - t, "secs to train for 5 epochs.")    # loss: 1.985 in 581 secs

# g = build_multilayer_graph_with_custom_cell(cell_type='Custom', num_steps=30)
# t = time.time()
# train_network(g, 5, num_steps=30)
# print("It took", time.time() - t, "secs to train for 5 epochs.")    # TODO Variable does not exist issue. loss: in secs
# # .. custom cell took longer to train and seems to perform worse than a standard GRU

# add dropout
# just add one line before

