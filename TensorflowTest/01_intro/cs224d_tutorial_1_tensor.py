import tensorflow as tf

tf.InteractiveSession()

a = tf.zeros((2,2)); b = tf.ones((2,2))

# Note: TensorFlow computations define a computation graph that has no numerical value until evaluated!
print tf.reduce_sum(b, reduction_indices=1).eval()
# [ 2.  2.]

print a.get_shape()
# (2, 2)

print tf.reshape(a, (1,4)).eval()
# [[ 0.  0.  0.  0.]]
