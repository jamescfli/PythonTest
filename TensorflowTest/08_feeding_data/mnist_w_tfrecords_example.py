import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import random

dataset_path = './csv/'
test_labels_file = "test-labels.csv"
train_labels_file = "train-labels.csv"


# We are not going to train a model in this example so if you want to train e.g. a neural network
# you probably want to do some one-hot encoding here.
def encode_label(label):
    return int(label)


def read_label_file(filename):
    filepaths = []
    labels = []
    for line in file(filename):
        filepath, label = line.split(',')
        filepaths.append(filepath)
        labels.append(encode_label(label))
    return filepaths, labels

# reading labels and file path
train_filepaths, train_labels = read_label_file(dataset_path + train_labels_file)
test_filepaths, test_labels = read_label_file(dataset_path + test_labels_file)

# transform relative path into full path
train_filepaths = [ dataset_path + fp for fp in train_filepaths]
test_filepaths = [ dataset_path + fp for fp in test_filepaths]

# for this example we will create or own test partition
all_filepaths = train_filepaths + test_filepaths
all_labels = train_labels + test_labels

# we limit the number of files to 20 to make the output more clear!
all_filepaths = all_filepaths[:20]
all_labels = all_labels[:20]

# convert to tensors
all_images = ops.convert_to_tensor(all_filepaths, dtype=dtypes.string)
all_labels = ops.convert_to_tensor(all_labels, dtype=dtypes.int32)

# define the test_set_size to be 5 samples, on the fly
# create a partition vector
partitions = [0] * len(all_filepaths)
test_set_size = 5
partitions[:test_set_size] = [1] * test_set_size
random.shuffle(partitions)
print(partitions)

# partition our data into a test and train set according to our partition vector
train_images, test_images = tf.dynamic_partition(all_images, partitions, num_partitions=2)
train_labels, test_labels = tf.dynamic_partition(all_labels, partitions, num_partitions=2)
# .. tf.dynamic_partition: partitions data into num_partitions tensors using indices from partitions

# tf.train.slice_input_producer: produces a slice of each Tensor in tensor_list
# Implemented using a Queue -- a QueueRunner for the Queue, string_input_producer
train_input_queue = tf.train.slice_input_producer(
    [train_images, train_labels],
    shuffle=False,
)
test_input_queue = tf.train.slice_input_producer(
    [test_images, test_labels],
    shuffle=False,
)

# process path and string tensor into an image and a label
file_content = tf.read_file(train_input_queue[0])
train_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
train_label = train_input_queue[1]

file_content = tf.read_file(test_input_queue[0])
test_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
test_label = test_input_queue[1]

# group samples to batch
# To use tf.train_batch we need to define the shape of our image tensors
# before they can be combined into batches
IMAGE_HEIGHT  = 28
IMAGE_WIDTH   = 28
NUM_CHANNELS  = 3
BATCH_SIZE    = 5
# define tensor shape
train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])


# collect batches of images before processing
train_image_batch, train_label_batch = tf.train.batch(  # or shuffle_batch
                                    [train_image, train_label],
                                    batch_size=BATCH_SIZE
                                    #,num_threads=1
                                    )
test_image_batch, test_label_batch = tf.train.batch(
                                    [test_image, test_label],
                                    batch_size=BATCH_SIZE
                                    #,num_threads=1
                                    )

with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print 'from the train set:'
    for i in range(20):
        print sess.run(train_label_batch)

    print "from the test set:"
    for i in range(10):
        print sess.run(test_label_batch)

    coord.request_stop()
    coord.join(threads)
    sess.close()