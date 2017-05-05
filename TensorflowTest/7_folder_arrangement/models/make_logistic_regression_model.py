import numpy as np
from basic_model import BasicAgent
import tensorflow as tf

# import MNIST data
from tensorflow.examples.tutorials.mnist import input_data


class LogisticRegression(BasicAgent):
    def __init__(self, config):
        super(LogisticRegression, self).__init__(config=config)
        # prepare data generator
        self.mnist = input_data.read_data_sets("../../basics/MNIST/MNIST_data", one_hot=True)

    def get_random_config(self, fixed_params={}):
        # static, because you want to be able to pass this to other processes
        # so they can independently generate random config of the current model
        # do not forget to use np.random.seed(config['random_seed'])
        pass

    def build_graph(self, graph):
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        self.W = tf.Variable(tf.zeros([784, 10]), name="weight")
        self.b = tf.Variable(tf.zeros([10]), name="bias")
        self.pred = tf.nn.softmax(tf.matmul(self.X, self.W) + self.b)
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.Y*tf.log(self.pred), reduction_indices=1))

        # set global step tensor, refer to the number of batches seen by the graph
        #   1718 in this case for one epoch
        # get the global_step value using tf.train.global_step()
        self.global_step_t = tf.Variable(0, trainable=False, name='global_step_t')

        self.optimizer = tf.train.GradientDescentOptimizer(self.config['lr'])\
            .minimize(self.cost, global_step=self.global_step_t)
        return graph

    def infer(self):
        avg_cost = 0
        nb_total_batch = int(self.mnist.train.num_examples/self.config['bsize'])

        for i in range(nb_total_batch):
            batch_xs, batch_ys = self.mnist.train.next_batch(self.config['bsize'])
            c = self.sess.run(self.cost, feed_dict={self.X: batch_xs, self.Y: batch_ys})
            avg_cost += c/nb_total_batch
        training_cost = avg_cost
        print "Training cost =", training_cost, '\n'

    def learn_from_epoch(self):
        # separate func to train per epoch and func to train globally
        avg_cost = 0
        # 55000/32 = 1718.75
        nb_total_batch = int(self.mnist.train.num_examples/self.config['bsize'])
        for i in range(nb_total_batch):
            batch_xs, batch_ys = self.mnist.train.next_batch(self.config['bsize'])
            _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.X: batch_xs, self.Y: batch_ys})
            avg_cost += c/nb_total_batch
        if self.config['debug']:
            print "\t >> cost=", "{:.9f}".format(avg_cost)


def make_model(config):
    return LogisticRegression(config=config)
