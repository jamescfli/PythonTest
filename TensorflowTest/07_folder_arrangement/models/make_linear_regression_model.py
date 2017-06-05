import numpy as np
from basic_model import BasicAgent
import tensorflow as tf


class LinearRegression(BasicAgent):
    def __init__(self, config):
        # prepare data generator
        np.random.seed(config['random_seed'])
        self.x_data = np.random.rand(100).astype(np.float32)
        noise = np.random.normal(scale=0.01, size=len(self.x_data))
        self.y_data = self.x_data * 0.1 + 0.3 + noise
        super(LinearRegression, self).__init__(config=config)

    def get_random_config(self, fixed_params={}):
        # static, because you want to be able to pass this to other processes
        # so they can independently generate random config of the current model
        pass

    def build_graph(self, graph):
        self.W = tf.Variable(tf.random_uniform([1], 0.0, 1.0), name="weight")
        self.b = tf.Variable(tf.zeros([1]), name="bias")
        self.pred = tf.add(tf.multiply(self.x_data, self.W), self.b)
        self.cost = tf.reduce_mean(tf.square(self.pred-self.y_data))

        self.global_step_t = tf.Variable(0, trainable=False, name='global_step_t')

        self.optimizer = tf.train.GradientDescentOptimizer(self.config['lr'])\
            .minimize(self.cost, global_step=self.global_step_t)
        return graph

    def tf_summary(self):
        self.sw = tf.summary.FileWriter(self.result_dir, self.sess.graph)

    def infer(self):
        testing_cost = self.sess.run(self.cost)
        print "Testing cost =", testing_cost, " W =", self.sess.run(self.W), " b =", self.sess.run(self.b), '\n'

    def learn_from_epoch(self):
        # separate func to train per epoch and func to train globally
        self.sess.run(self.optimizer)
        # self.sess.run(self.optimizer, feed_dict={self.X: train_X, self.Y: train_Y})

        if self.config['debug']:
            print "\t >> cost=", "{:.9f}".format(self.sess.run(self.cost)), \
                " W =", self.sess.run(self.W), " b =", self.sess.run(self.b)


def make_model(config):
    return LinearRegression(config=config)
