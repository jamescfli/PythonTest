import tensorflow as tf

with tf.Session():
    input = tf.placeholder(tf.float32)
    classifier = tf.pow(input, tf.constant(2, dtype=tf.float32))
    print(classifier.eval(feed_dict={input: 3}))    # 9.0
    print(classifier.eval())
    # .. InvalidArgumentError (see above for traceback):
    # .. You must feed a value for placeholder tensor 'Placeholder' with dtype float
