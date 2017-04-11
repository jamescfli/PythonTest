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

# build a custom RNN cell
class GRUCell(tf.contrib.rnn.core_rnn_cell.RNNCell):
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
                ru = tf.contrib.rnn.core_rnn_cell._linear([inputs, state], 2*self._num_units, True, 1.0)
                ru = tf.nn.sigmoid(ru)
                r, u = tf.split(ru, 2, 1)
            with tf.variable_scope("Candidate"):
                c = tf.nn.tanh(tf.nn.rnn_cell_impl._linear([inputs, r*state], self._num_units, True))
            new_h = u * state + (1-u) * c
        return new_h, new_h


# Layer normalization
