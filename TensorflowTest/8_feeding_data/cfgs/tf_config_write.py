import tensorflow as tf


tf.app.flags.DEFINE_string('directory', '/tmp/data',
                           'Directory to download data files andg write the converted result')
tf.app.flags.DEFINE_integer('validation_size', 5000,
                            'Number of examples to seperate from the training data for validation')
FLAGS = tf.app.flags.FLAGS
