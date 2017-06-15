import tensorflow as tf


tf.app.flags.DEFINE_float('learning_rate', 0.01,
                          'Initial learning rate')
tf.app.flags.DEFINE_integer('nb_epochs', 10,
                            'Number of epochs to run trainer')
tf.app.flags.DEFINE_integer('hidden1', 128,
                            'Number of hidden units in layer 1')
tf.app.flags.DEFINE_integer('hidden2', 32,
                            'Number of hidden units in layer 2')
tf.app.flags.DEFINE_integer('batch_size', 100,
                            'Batch size')
tf.app.flags.DEFINE_string('train_dir', '/tmp/data',
                           'Directory with the training data in tfrecords format')
tf.app.flags.DEFINE_integer('validation_size', 5000,
                            'Number of examples to seperate from the training data for validation')
FLAGS = tf.app.flags.FLAGS
