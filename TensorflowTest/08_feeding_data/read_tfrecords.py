#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.contrib.learn.python.learn.datasets import mnist as mnist_datasets
import cfgs.tf_config_read as cfg
import numpy as np

# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'valid.tfrecords'     # where validation size = 5000


def read_and_decode(filename_queue):    # a list of filenames
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
    filename = os.path.join(cfg.FLAGS.train_dir, TRAIN_FILE if train else VALIDATION_FILE)
    print(filename)
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
        # min_after_dequeue:
        #     Minimum number elements/samples in the queue after a dequeue,
        #     used to ensure a level of mixing (shuffling) of elements.
        #     in mnist example validation set has 5000 samples, so min_after_dequeue = 1000
        # capacity: An integer. The maximum number of elements in the queue.
        # Steps:
        #   1) determine min_after_dequeue first, big -> more fair shuffle but slow start and large memory consumption
        #   2) decide capacity according to min_after_dequeue + (num_threads + a small safety margin) * batch_size
        return images, sparse_labels    # output tensors


def run_training():
    with tf.Graph().as_default():
        # train data and run valid after each epoch, so nb_epochs=1
        images, labels = inputs(train=True, batch_size=cfg.FLAGS.batch_size, nb_epochs=cfg.FLAGS.nb_epochs)
        logits = mnist.inference(images, cfg.FLAGS.hidden1, cfg.FLAGS.hidden2)
        loss = mnist.loss(logits, labels)

        train_op = mnist.training(loss, cfg.FLAGS.learning_rate)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess = tf.Session()
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        data_sets = mnist_datasets.read_data_sets(cfg.FLAGS.train_dir,
                                                  dtype=tf.uint8,
                                                  reshape=False,
                                                  validation_size=cfg.FLAGS.validation_size)

        nb_train_samples = data_sets.train.num_examples
        # print('training samples: {}; batch_size: {}'.format(nb_train_samples, cfg.FLAGS.batch_size))
        # .. 55000 and 100

        # prepare validation data in terms of tf.constant
        image_valid_np = data_sets.validation.images.reshape((cfg.FLAGS.validation_size, mnist.IMAGE_PIXELS))
        label_valid_np = data_sets.validation.labels        # shape (5000,)
        # to fit the batch size
        idx_valid = np.random.choice(cfg.FLAGS.validation_size, cfg.FLAGS.batch_size, replace=False)
        image_valid_np = image_valid_np[idx_valid, :]
        image_valid_np = image_valid_np * (1. / 255) - 0.5      # remember to preprocessing
        label_valid_np = label_valid_np[idx_valid]

        step = 0
        epoch_idx = 0
        try:
            start_time = time.time()
            while not coord.should_stop():
                _, loss_value = sess.run([train_op, loss])
                step += 1
                if step >= nb_train_samples // cfg.FLAGS.batch_size:
                    epoch_idx += 1
                    end_time = time.time()
                    duration = end_time - start_time
                    print('Training Epoch {}, Step {}: loss = {:.02f} ({:.03f} sec)'
                          .format(epoch_idx, step, loss_value, duration))
                    start_time = end_time   # re-timing
                    step = 0                # reset step counter
                    # derive loss on validation dataset
                    loss_valid_value = sess.run(loss, feed_dict={images: image_valid_np, labels: label_valid_np})
                    print('Validation Epoch {}: loss = {:.02f}'
                          .format(epoch_idx, loss_valid_value))
        except tf.errors.OutOfRangeError:
            print('Done training for epoch {}, {} steps'.format(epoch_idx, step))
        finally:
            coord.request_stop()



        # # restart runner for validation data
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #
        # step = 0
        # try:
        #     start_time = time.time()
        #     while not coord.should_stop():
        #         loss_value_valid = sess.run(loss_valid)
        #         step += 1
        # except tf.errors.OutOfRangeError:
        #     print('Done validation for epoch {}, {} steps'.format(epoch_idx, step))
        # finally:
        #     coord.request_stop()
        #     duration = time.time() - start_time
        #     print('Validation: Epoch {}, Step {}: loss = {:.02f} ({:.03f} sec)'
        #           .format(epoch_idx, step, loss_value_valid, duration))

        coord.join(threads)
        sess.close()


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()


# Training Epoch 1, Step 550: loss = 0.85 (13.488 sec)
# Training Epoch 2, Step 550: loss = 0.48 (13.618 sec)
# Training Epoch 3, Step 550: loss = 0.29 (13.152 sec)
# Training Epoch 4, Step 550: loss = 0.39 (13.182 sec)
# Training Epoch 5, Step 550: loss = 0.28 (13.163 sec)
# Training Epoch 6, Step 550: loss = 0.46 (13.192 sec)
# Training Epoch 7, Step 550: loss = 0.22 (13.184 sec)
# Training Epoch 8, Step 550: loss = 0.28 (13.140 sec)
# Training Epoch 9, Step 550: loss = 0.37 (13.207 sec)
# Training Epoch 10, Step 550: loss = 0.14 (12.927 sec)
# Done training for epoch 10, 0 steps

# Training Epoch 1, Step 550: loss = 0.71 (13.381 sec)
# Validation Epoch 1: loss = 0.84
# Training Epoch 2, Step 550: loss = 0.44 (13.176 sec)
# Validation Epoch 2: loss = 0.49
# Training Epoch 3, Step 550: loss = 0.26 (13.450 sec)
# Validation Epoch 3: loss = 0.38
# Training Epoch 4, Step 550: loss = 0.36 (13.722 sec)
# Validation Epoch 4: loss = 0.35
# Training Epoch 5, Step 550: loss = 0.21 (13.440 sec)
# Validation Epoch 5: loss = 0.32
# Training Epoch 6, Step 550: loss = 0.34 (13.106 sec)
# Validation Epoch 6: loss = 0.30
# Training Epoch 7, Step 550: loss = 0.31 (13.162 sec)
# Validation Epoch 7: loss = 0.29
# Training Epoch 8, Step 550: loss = 0.24 (13.075 sec)
# Validation Epoch 8: loss = 0.28
# Training Epoch 9, Step 550: loss = 0.25 (13.092 sec)
# Validation Epoch 9: loss = 0.26
# Training Epoch 10, Step 550: loss = 0.14 (12.871 sec)
# Validation Epoch 10: loss = 0.24
# Done training for epoch 10, 0 steps
