#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import mnist

# Basic model parameters as external flags.
FLAGS = None

# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'valid.tfrecords'


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                       })   # output Tensor
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    # .. others: tf.image.decode_jpeg, tf.image.decode_png, tf.image.decode_image
    image.set_shape([mnist.IMAGE_PIXELS])   # image.shape = (784,)
    # convert to [-0.5, +0.5]
    image = tf.cast(image, tf.float32) * (1./255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return image, label


def inputs(train, batch_size, nb_epochs):
    # Note that an tf.train.QueueRunner is added to the graph, which
    # must be run using e.g. tf.train.start_queue_runners().
    if not nb_epochs: nb_epochs = None
    filename = os.path.join(FLAGS.train_dir, TRAIN_FILE if train else VALIDATION_FILE)
    # .. could be with multiple files
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename], num_epochs=nb_epochs)   # output a FIFOQuue
        image, label = read_and_decode(filename_queue)  # output two tensors
        # shuffle the examples and collect them into batch_size batches
        # internally, uses a RandomShuffleQueue
        # we run this in two threads to avoid being a bottleneck
        images, sparse_labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=2,
            capacity=1000 + 3 * batch_size,     # 3 = 2 threads + 1 margin
            # ensures a minimum amount of shuffling of examples
            min_after_dequeue=1000
        )
        return images, sparse_labels    # output tensors


def run_training():
    with tf.Graph().as_default():
        images, labels = inputs(train=True, batch_size=FLAGS.batch_size, nb_epochs=FLAGS.nb_epochs)
        logits = mnist.inference(images, FLAGS.hidden1, FLAGS.hidden2)
        loss = mnist.loss(logits, labels)
        train_op = mnist.training(loss, FLAGS.learning_rate)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess = tf.Session()
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # Thread 6~10, 5 on MBP

        try:
            step = 0
            while not coord.should_stop():
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss])
                duration = time.time() - start_time
                if step % 100 == 0:
                    print('Step {}: loss = {:02f} ({:03f} sec)'.format(step, loss_value, duration))
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for {} epochs, {} steps'.format(FLAGS.nb_epochs, step))
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()


def main(_):
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--nb_epochs',
                        type=int,
                        default=2,
                        help='Number of epochs to run trainer')
    parser.add_argument('--hidden1',
                        type=int,
                        default=128,
                        help='Number of hidden units in layer 1')
    parser.add_argument('--hidden2',
                        type=int,
                        default=32,
                        help='Number of hidden units in layer 2')
    parser.add_argument('--batch_size',
                        type=int,
                        default=100,
                        help='Batch size')
    parser.add_argument('--train_dir',
                        type=str,
                        default='/tmp/data',
                        help='Directory with the training data in tfrecords format')
    FLAGS, unparsed = parser.parse_known_args()     # unparsed = []
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

