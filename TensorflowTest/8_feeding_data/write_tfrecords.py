# Converts MNIST data to TFRecords

# absolute_import does not care about whether something is part of the standard library,
# and import string will not always give you the standard-library module with absolute imports on.
# from __future__ import absolute_import means that if you import string,
# Python will always look for a top-level string module, rather than current_package.string
from __future__ import absolute_import
from __future__ import division     # print(1/2) = 0.5
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import mnist

FLAGS = None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecords(data_set, name):
    images = data_set.images
    labels = data_set.labels
    num_examples = data_set.num_examples

    if images.shape[0] != num_examples:
        raise ValueError('Images size {} does not match label size {}.'.format(images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    print('Writing {}'.format(filename))
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)
        }))
        writer.write(example.SerializeToString())
    writer.close()


def main(_):        # _ : unused_argv
    data_sets = mnist.read_data_sets(FLAGS.directory,
                                     dtype=tf.uint8,
                                     reshape=False,
                                     validation_size=FLAGS.validation_size)
    convert_to_tfrecords(data_sets.train, 'train')
    convert_to_tfrecords(data_sets.validation, 'valid')
    convert_to_tfrecords(data_sets.test, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',
                        type=str,
                        default='/tmp/data',
                        help='Directory to download data files andg write the converted result')
    parser.add_argument('--validation_size',
                        type=int,
                        default=5000,
                        help='Number of examples to seperate from the training data for validation')
    FLAGS, unparsed = parser.parse_known_args()
    # It's just a very quick wrapper that handles flag parsing and then dispatches to your own main
    tf.app.run(argv=[sys.argv[0]] + unparsed)
