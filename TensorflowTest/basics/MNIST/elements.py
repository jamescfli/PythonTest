import tensorflow as tf


graph = tf.Graph()      # use default if without writing this
with graph.as_default():    # with - hook methods: __enter__() and __exit__()
    foo = tf.Variable(3, name='foo')    # name is important when importing weights
    bar = tf.Variable(2, name='bar')
    result = foo + bar
    initialize = tf.global_variables_initializer()

print(result)   # Tensor("add:0", shape=(), dtype=int32), no operation

with tf.Session(graph=graph) as sess:   # prevent session occupies mem afterwards
    sess.run(initialize)    # real init
    res = sess.run(result)  # and run
    print(res)

# Variables and Constants
weights = tf.Variable(tf.random_normal((784, 200), stddev=0.35, name='weights'))
biases = tf.Variable(tf.zeros((200), name='biases'))
# .. initialized by tf.assign

# apply global_variables_initializer() to initialize all
# if only initialize part of parameters, the following will do so
init_ab = tf.variables_initializer([a, b], name='init_ab')
# present value of Variables by eval()
