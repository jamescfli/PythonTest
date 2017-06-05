import numpy as np
import tensorflow as tf

import plot_boundary_on_data  

# Global variables.
NUM_LABELS = 2    # The number of labels.
BATCH_SIZE = 100   # The number of training examples to use per training step.

# Define the flags useable from the command line.
tf.app.flags.DEFINE_string('train', 'simdata/saturn_data_train.csv',
                           'File containing the training data (labels & features).')
tf.app.flags.DEFINE_string('test', 'simdata/saturn_data_eval.csv',
                           'File containing the test data (labels & features).')
tf.app.flags.DEFINE_integer('num_epochs', 5,
                            'Number of examples to separate from the training '
                            'data for the validation set.')
tf.app.flags.DEFINE_boolean('verbose', True, 'Produce verbose output.')
tf.app.flags.DEFINE_boolean('plot', True, 'Plot the final decision boundary on the data.')
FLAGS = tf.app.flags.FLAGS


# Extract numpy representations of the labels and features given rows consisting of:
#   label, feat_0, feat_1, ..., feat_n
def extract_data(filename):

    # Arrays to hold the labels and feature vectors.
    labels = []
    fvecs = []

    # Iterate over the rows, splitting the label from the features. Convert labels
    # to integers and features to floats.
    for line in file(filename):
        row = line.split(",")
        labels.append(int(row[0]))                  # label goes first
        fvecs.append([float(x) for x in row[1:]])   # then features

    # Convert the array of float arrays into a numpy float matrix.
    fvecs_np = np.array(fvecs).astype(np.float32)

    # Convert the array of int labels into a numpy array.
    labels_np = np.array(labels).astype(dtype=np.uint8)

    # Convert the int numpy array into a one-hot matrix. smart
    labels_onehot = (np.arange(NUM_LABELS) == labels_np[:, None]).astype(np.float32)
    # .. labels_np[:, None].shape = (1000,1); labels_np.shape = (1000,)

    # Return a pair of the feature matrix and the one-hot label matrix.
    return fvecs_np, labels_onehot


def main(_):
    # Be verbose?
    verbose = FLAGS.verbose

    # Plot? 
    plot = FLAGS.plot
    
    # Get the data.
    train_data_filename = FLAGS.train
    test_data_filename = FLAGS.test

    # Extract it into numpy matrices.
    train_data, train_labels = extract_data(train_data_filename)    # label with one hot vector
    test_data, test_labels = extract_data(test_data_filename)

    # Get the shape of the training data.
    train_size, num_features = train_data.shape

    # Get the number of epochs for training.
    num_epochs = FLAGS.num_epochs

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    x = tf.placeholder("float", shape=[None, num_features])
    y_ = tf.placeholder("float", shape=[None, NUM_LABELS])
    
    # These are the weights that inform how much each feature contributes to
    # the classification.
    W = tf.Variable(tf.zeros([num_features, NUM_LABELS]))
    b = tf.Variable(tf.zeros([NUM_LABELS]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # Optimization.
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # For the test data, hold the entire dataset in one constant node.
    test_data_node = tf.constant(test_data)

    # Evaluation.
    predicted_class = tf.argmax(y,1)        # for not integer case
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))      # either 0 or 1 for the max index
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # all tensors were ready

    # Create a local session to run this computation.
    with tf.Session() as sess:

        # Run all the initializers to prepare the trainable parameters.
        sess.run(tf.global_variables_initializer())
        # tf.global_variables_initializer().run()
        
        # Iterate and train.
        for step in xrange(num_epochs * train_size // BATCH_SIZE):
    
            offset = (step * BATCH_SIZE) % train_size

            # get a batch of data
            batch_data = train_data[offset:(offset + BATCH_SIZE), :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]

            # feed data into the model
            train_step.run(feed_dict={x: batch_data, y_: batch_labels})

            if verbose:
                if (step+1) % 5 == 0:
                    print 'Accuracy @ step {:02d}: {}'\
                        .format(step, accuracy.eval(feed_dict={x: test_data, y_: test_labels}))

        # Give very detailed output.
        if verbose:
            print
            print 'Weight matrix.'
            print sess.run(W)
            print
            print 'Bias vector.'
            print sess.run(b)
            print
            print "Applying model to first test instance."
            first = test_data[:1]
            print "Point =", first
            print "Wx+b = ", sess.run(tf.matmul(first,W)+b)
            print "softmax(Wx+b) = ", sess.run(tf.nn.softmax(tf.matmul(first,W)+b))
            print
            
        print "Accuracy:", accuracy.eval(feed_dict={x: test_data, y_: test_labels})

        if plot:
            eval_fun = lambda X: predicted_class.eval(feed_dict={x:X}); 
            plot_boundary_on_data.plot(test_data, test_labels, eval_fun)
            plot_boundary_on_data.plot(train_data, train_labels, eval_fun)
    
if __name__ == '__main__':
    tf.app.run()
