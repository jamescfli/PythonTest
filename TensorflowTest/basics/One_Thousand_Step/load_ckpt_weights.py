import tensorflow as tf


with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# layer 1
#                inputs, in_size, out_size, act_func = None
# l1 = add_layer(xs, 1, 10, act_func=tf.nn.relu)
with tf.name_scope('layer'):
    with tf.name_scope('weights'):
        weights1 = tf.Variable(tf.random_normal([1, 10]), name='w')
    with tf.name_scope('biases'):
        biases1 = tf.Variable(tf.zeros([1, 10]) + 0.1, name='b')
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(xs, weights1) + biases1
    l1 = tf.nn.relu(wx_plus_b)
# layer 2: output
# prediction = add_layer(l1, 10, 1, act_func=None)
with tf.name_scope('layer'):
    with tf.name_scope('weights'):
        weights2 = tf.Variable(tf.random_normal([10, 1]), name='w')
    with tf.name_scope('biases'):
        biases2 = tf.Variable(tf.zeros([1, 1]) + 0.1, name='b')
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(l1, weights2) + biases2
    prediction = wx_plus_b

# define loss
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

# optimizer
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# initialize
init = tf.initialize_all_variables()

# saver
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, './saver/save_net.ckpt')
    print("weights1: ", sess.run(weights1))
    print("biases1: ", sess.run(biases1))
    print("weights2: ", sess.run(weights2))
    print("biases2: ", sess.run(biases2))
    # ('weights1: ', array([[  5.83120704e-01,   8.99606869e-02,  -1.81712874e-03,
    #          -4.08732183e-02,   5.05685329e-01,  -3.49428803e-01,
    #          -4.94837254e-01,   3.87535036e-01,  -1.84821665e+00,
    #           1.80748677e+00]], dtype=float32))
    # ('biases1: ', array([[ 0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1]], dtype=float32))
    # ('weights2: ', array([[ 0.20100932],
    #        [-0.21578397],
    #        [-0.62004727],
    #        [-0.0956822 ],
    #        [ 0.67681986],
    #        [-0.78813452],
    #        [ 0.52790219],
    #        [ 0.91009116],
    #        [ 0.20386617],
    #        [-0.98729849]], dtype=float32))
    # ('biases2: ', array([[ 0.1]], dtype=float32))
