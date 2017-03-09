import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, act_func = None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal([in_size, out_size]), name='w')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        with tf.name_scope('wx_plus_b'):
            wx_plus_b = tf.matmul(inputs, weights) + biases
        if act_func is None:
            outputs = wx_plus_b
        else:
            outputs = act_func(wx_plus_b)
    return outputs


# prepare data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]     # shape (300, 1)
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# layer 1
l1 = add_layer(xs, 1, 10, act_func=tf.nn.relu)
# layer 2: output
prediction = add_layer(l1, 10, 1, act_func=None)

# define loss
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

# optimizer
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# initialize
init = tf.initialize_all_variables()

# saver
saver = tf.train.Saver()

# log_path
logs_path = './tmp/tf_logs'

# with tf.Session() as sess:
#     # init
#     sess.run(init)
#     save_path = saver.save(sess, './saver/save_net.ckpt')
#     print('Save to path: ', save_path)
#     summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
#     for i in range(1000):
#         summary = sess.run(optimizer, feed_dict={xs: x_data, ys: y_data})
#         summary_writer.add_summary(summary, i)
#         if i % 50 == 0:
#             print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# # Create a summary to monitor accuracy tensor
# tf.summary.scalar("accuracy", acc)
# summary merged, merged the above
merged_summary = tf.summary.merge_all()

with tf.Session() as sess:
    # init
    sess.run(init)
    save_path = saver.save(sess, './saver/save_net.ckpt')
    print('Save to path: ', save_path)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    # .. or summary_writer.add_graph(sess.graph) if not defined in the last line
    for i in range(1000):
        # _, cost, summary = sess.run([optimizer, loss, merged_summary], feed_dict={xs: x_data, ys: y_data})
        sess.run(optimizer, feed_dict={xs: x_data, ys: y_data})
        # print('Iter #{}:{}'.format(i, cost))
        # summary_writer.add_summary(summary, i)
        if i % 50 == 0:
            cost, summary = sess.run([loss, merged_summary], feed_dict={xs: x_data, ys: y_data})
            print('Iter #{}:{}'.format(i, cost))
            summary_writer.add_summary(summary, i)  # write summary and iter # to file

# visualize by tensorboard --logdir='./tmp/tf_logs'