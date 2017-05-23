from basic_model import BasicAgent
import tensorflow as tf
import numpy as np

# import MNIST data
from tensorflow.examples.tutorials.mnist import input_data


class NearestNeighbour(BasicAgent):
    def __init__(self, config):
        super(NearestNeighbour, self).__init__(config=config)
        # prepare data generator
        self.mnist = input_data.read_data_sets("../../basics/MNIST/MNIST_data", one_hot=True)
        self.Xtr, self.Ytr = self.mnist.train.next_batch(5000)  # (5000, 784) (5000, 10)
        self.Xte, self.Yte = self.mnist.test.next_batch(200)    # (200, 784) (200, 10)

    def get_random_config(self, fixed_params={}):
        # static, because you want to be able to pass this to other processes
        # so they can independently generate random config of the current model
        # do not forget to use np.random.seed(config['random_seed'])
        pass

    def build_graph(self, graph):
        self.xtr = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        self.xte = tf.placeholder(dtype=tf.float32, shape=[784])    # one vector compares with all in self.xtr
        self.distance = tf.reduce_sum(tf.abs(tf.add(self.xtr, tf.negative(self.xte))), reduction_indices=1)
        self.pred = tf.argmin(self.distance, 0)

        self.global_step_t = tf.Variable(0, trainable=False, name='global_step_t')

        return graph

    def infer(self):
        accuracy = 0.

        for i in range(len(self.Xte)):
            # get nn
            nn_index = self.sess.run(self.pred, feed_dict={self.xtr: self.Xtr, self.xte: self.Xte[i, :]})
            # get nn class label and compare to true label
            print("Test", i, "Prediction:", np.argmax(self.Ytr[nn_index]), "True class:", np.argmax(self.Yte[i]))
            # calculate accuracy
            if np.argmax(self.Ytr[nn_index]) == np.argmax(self.Yte[i]):
                accuracy += 1./len(self.Xte)
        print("Done!")
        print("Accuracy:", accuracy)    # 'Accuracy:', 0.92

    def learn_from_epoch(self):
        print("No trainable parameters")
        pass    # no training for NN - NearestNeighbour


def make_model(config):
    return NearestNeighbour(config=config)
