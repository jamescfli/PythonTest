import tensorflow as tf
import numpy as np

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# compute lenght
def length(sequence):
    # collapse into scalar by max
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))    # sign function +/-1 for x </> 0
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

# use length info
max_length = 100
frame_size = 64
nb_hidden = 200

sequence = tf.placeholder(tf.float32, [None, max_length, frame_size])
output, state = tf.nn.dynamic_rnn(
    tf.contrib.rnn.core_rnn_cell.GRUCell(nb_hidden),
    sequence,
    dtype=tf.float32,
    sequence_length=length(sequence)
)

# mask the cost function, mask out used frames and compute mean by dividing actual length
def cost(output, target):
    # compute cross entropy for each frame
    cross_entropy = target * tf.log(output)
    cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
    mask = tf.sign(tf.reduce_max(tf.abs(target), reduction_indices=2))
    cross_entropy *= mask
    # average over actual length
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
    cross_entropy /= tf.reduce_sum(mask, reduction_indices=1)
    return tf.reduce_mean(cross_entropy)

# select last relevant output
def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])   # flatten output to shape frames in all examples x output size
    relevant = tf.gather(flat, index)   # perform the actual indexing
    return relevant


nb_classes = 10

last = last_relevant(output)
weight = tf.Variable(tf.truncated_normal([nb_hidden, nb_classes], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[nb_classes]))
prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

# github code: https://gist.github.com/danijar/3f3b547ff68effb03e20c470af22c696

