import tensorflow as tf


# define
foo = tf.placeholder(tf.int32, shape=[1], name='foo')
bar = tf.constant(2, name='bar')
result = foo + bar

with tf.Session() as sess:
    # print(sess.run(result))     # InvalidArgumentError: You must feed a value for placeholder tensor 'foo'
    print(sess.run(result, {foo:[3]}))  # [5]
