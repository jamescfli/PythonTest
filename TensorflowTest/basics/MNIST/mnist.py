from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


# IOError: [Errno socket error] [Errno 60] Operation timed out
#   http://yann.lecun.com/exdb/mnist/ is down
#   use https://s3.amazonaws.com/lasagne/recipes/datasets/mnist/ instead
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])     # None will be the batch size
y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# define model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# define loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)
init = tf.global_variables_initializer()

# run
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
    # verify
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))   # 0.9168
