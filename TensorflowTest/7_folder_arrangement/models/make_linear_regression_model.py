import numpy as np
from basic_model import BasicAgent
import tensorflow as tf

rng = np.random

train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                      7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                      2.827,3.465,1.65,2.904,2.42,2.94,1.3])
nb_sample = train_X.shape[0]
# train_X = train_X[:, np.newaxis]
# train_Y = train_Y[:, np.newaxis]


class LinearRegression(BasicAgent):
    # def __init__(self, config):
    #     self.init_op = tf.global_variables_initializer()
    #     super(LinearRegression, self).__init__(config=config)

    def get_random_config(self, fixed_params={}):
        # static, because you want to be able to pass this to other processes
        # so they can independently generate random config of the current model
        pass

    def build_graph(self, graph):
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, 1])      # TODO issue unresolved for scalar input
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        # self.X = tf.placeholder(dtype=tf.float32)
        # self.Y = tf.placeholder(dtype=tf.float32)
        # self.W = tf.Variable(rng.randn(), name="weight")
        # self.b = tf.Variable(rng.randn(), name="bias")
        self.W = tf.Variable(tf.zeros([1, 1]), name="weight")
        self.b = tf.Variable(tf.zeros([1]), name="bias")
        self.pred = tf.add(tf.multiply(self.X, self.W), self.b)
        self.cost = tf.reduce_mean(tf.square(self.pred-self.Y))
        self.optimizer = tf.train.GradientDescentOptimizer(self.config['lr']).minimize(self.cost)
        return graph

    def infer(self):
        training_cost = self.sess.run(self.cost, feed_dict={self.X: train_X, self.Y: train_Y})
        print "Training cost =", training_cost, "W=", self.sess.run(self.W), "b=", self.sess.run(self.b), '\n'

    def learn_from_epoch(self):
        # separate func to train per epoch and func to train globally
        for (x, y) in zip(train_X, train_Y):
            self.sess.run(self.optimizer, feed_dict={self.X: x, self.Y: y})
        # self.sess.run(self.optimizer, feed_dict={self.X: train_X, self.Y: train_Y})

        if self.config['debug']:
            print "cost=", "{:.9f}".format(self.sess.run(self.cost)), \
                "W=", self.sess.run(self.W), "b=", self.sess.run(self.b)


# def reset_graph():
#     if 'sess' in globals() and sess:
#         sess.close()    # just in case
#     tf.reset_default_graph()


def make_model(config):
    # reset_graph()       # check whether this is necessary
    return LinearRegression(config=config)


