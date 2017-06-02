import tensorflow as tf
import numpy as np


a = tf.constant(5.0)
b = tf.constant(6.0)
c = a*b

with tf.Session() as sess:
    print(sess.run(c))
    print(c.eval())     # c.eval() is just syntactic sugar for sess.run(c) in the currently active session!


W1 = tf.ones((2,2))
W2 = tf.Variable(tf.zeros((2,2)), name="weights")

with tf.Session() as sess:
    print(sess.run(W1))
    sess.run(tf.global_variables_initializer())
    print(sess.run(W2))


W = tf.Variable(tf.zeros((2,2)), name="weights")
R = tf.Variable(tf.random_normal((2,2)), name="random_weights")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(W))
    print(sess.run(R))

state = tf.Variable(0, name='counter')
new_value = tf.add(state, tf.constant(1))
update = tf.assign(state, new_value)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

a = np.zeros((3,3))
ta = tf.convert_to_tensor(a)
with tf.Session() as sess:
    print(sess.run(ta))

# tf.convert_to_tensor() is convenient, but doesn't scale.
# Use tf.placeholder variables
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
# A feed_dict is a python dictionary mapping from tf.placeholder vars (or their names)
# to data (numpy arrays, lists, etc.).
with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))
# .. [array([ 14.], dtype=float32)]

# variable name scope
with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v = tf.get_variable("v", [1])
assert v.name == "foo/bar/v:0"
# and use reuse_variables() to implement RNNs
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
    tf.get_variable_scope().reuse_variables()
    v1 = tf.get_variable("v", [1])
assert v1 == v
# reuse set to False
with tf.variable_scope("foo"):
    # Create and return new variable
    v = tf.get_variable("v", [1])
assert v.name == "foo/v:0"
# reuse set to true
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
with tf.variable_scope("foo", reuse=True):
    # Search for existing variable with given name.
    v1 = tf.get_variable("v", [1])
assert v1 == v

