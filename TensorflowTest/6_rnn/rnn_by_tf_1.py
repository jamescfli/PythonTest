import numpy as np


# print("Expected cross entropy loss if the model:")
# print("- learns neither dependency:", -(0.625 * np.log(0.625) +
#                                       0.375 * np.log(0.375)))
# print("- learns first dependency:  ",
#       -0.5 * (0.875 * np.log(0.875) + 0.125 * np.log(0.125))
#       -0.5 * (0.625 * np.log(0.625) + 0.375 * np.log(0.375)))
# print("- learns both dependencies: ",
#       -0.50 * (0.75 * np.log(0.75) + 0.25 * np.log(0.25))
#       - 0.25 * (2 * 0.50 * np.log (0.50)) - 0.25 * (0))
#
# # Expected cross entropy loss if the model:
# # ('- learns neither dependency:', 0.66156323815798213)
# # ('- learns first dependency:  ', 0.51916669970720941)
# # ('- learns both dependencies: ', 0.4544543674493905)

import tensorflow as tf
import matplotlib.pyplot as plt


num_epochs = 10
num_steps = 10      # learn how many back steps
batch_size = 200
num_classes = 2
state_size = 16     # from 4 to 16 when num_steps is 10 > 8, for improving representing capability
learning_rate = 0.1


def gen_data(size=1000000):
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i-3] == 1:
            threshold += 0.5
        if X[i-8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)


def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length*i : batch_partition_length*(i+1)]
        data_y[i] = raw_y[batch_partition_length*i : batch_partition_length*(i+1)]
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i*num_steps : (i+1)*num_steps]
        y = data_y[:, i*num_steps : (i+1)*num_steps]
        yield (x, y)


def gen_epochs(n, num_steps):
    for i in range(n):
        yield gen_batch(gen_data(), batch_size, num_steps)


# build model
x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')
init_state = tf.zeros([batch_size, state_size])

# # static rnn
# # turn x placeholder to a list of one-hot tensors
# # rnn_inputs is a list of num_steps tensors with shape [batch_size, num_classes]
# x_one_hot = tf.one_hot(x, num_classes)
# # print np.array(x_one_hot).shape     # ()
# rnn_inputs = tf.unstack(x_one_hot, axis=1)
# # print np.array(rnn_inputs).shape    # (5,)

# dynamic rnn
rnn_inputs = tf.one_hot(x, num_classes)

# define rnn_cell
# # method 1)
# with tf.variable_scope('rnn_cell'):
#     W = tf.get_variable('W', [num_classes+state_size, state_size])
#     b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
#
#
# def rnn_cell(rnn_input, state):
#     with tf.variable_scope('rnn_cell', reuse=True):
#         W = tf.get_variable('W', [num_classes+state_size, state_size])
#         b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
#     # tf.concat(values, axis, name='concat')
#     return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)
#
# # add rnn_cells to graph
# # this is static_rnn. in practice, it is better to use dynamic_rnn instead
# state = init_state
# rnn_outputs = []
# for rnn_input in rnn_inputs:
#     state = rnn_cell(rnn_input, state)
#     rnn_outputs.append(state)
# final_state = rnn_outputs[-1]

# method 2) cleaner and simpler
cell = tf.contrib.rnn.BasicRNNCell(state_size)

# # static rnn
# rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=init_state)

# dynamic rnn
rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

# prediction, loss, training step
with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

# # static rnn
# logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
# predictions = [tf.nn.softmax(logit) for logit in logits]
# # turn our y placeholder into a list of labels
# y_as_list = tf.unstack(y, num=num_steps, axis=1)                        # ground truth
# losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit)
#           for logit, label in zip(logits, y_as_list)]

# dynamic rnn
logits = tf.reshape(tf.matmul(tf.reshape(rnn_outputs, [-1, state_size]), W) + b, [batch_size, num_steps, num_classes])
predictions = tf.nn.softmax(logits)
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

# train
def train_network(num_epochs, num_steps, state_size=4, verbose=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
            training_loss = 0
            training_state = np.zeros((batch_size, state_size))
            if verbose:
                print("\nEpoch", idx)
            for step, (X, Y) in enumerate(epoch):
                tr_losses, training_loss_, training_state, _ = \
                    sess.run([losses, total_loss, final_state, train_step],
                             feed_dict={x:X, y:Y, init_state:training_state})
                training_loss += training_loss_
                if step % 100 == 0 and step > 0:
                    if verbose:
                        # TODO figure out '250 steps'
                        print("Average loss at step", step, "for last 250 steps:", training_loss/100)
                        # print('num_steps:', num_steps)    # = 10
                    training_losses.append(training_loss/100)
                    training_loss = 0
    return training_losses

# training_losses = train_network(1, num_steps)
training_losses = train_network(num_epochs=num_epochs, num_steps=num_steps, state_size=state_size)
plt.plot(training_losses)
plt.show()
# .. network very quickly learns to capture the first dependency (but not the second)
# .. and expected cross-entropy loss of 0.52

# num_epochs = 10 -> (0.454)

# dynamic RNN: dynamically create the graph at execution time
# [batch_size, features] -> [batch_size, num_steps, features], i.e. num_steps can change during training