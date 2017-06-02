import tensorflow as tf
import numpy as np

# # 1) split
# value = tf.ones((5, 30), tf.int32)
# print np.array(value.shape)     # [Dimension(5) Dimension(30)]
#
# # 'value' is a tensor with shape [5, 30]
# # Split 'value' into 3 tensors with sizes [4, 15, 11] along dimension 1
# split0, split1, split2 = tf.split(value, [4, 15, 11], 1)
# print np.array(split0.shape)    # [Dimension(5) Dimension(4)]
# tf.shape(split1)                # ==> [5, 15]
# tf.shape(split2)                # ==> [5, 11]
# # Split 'value' into 3 tensors along dimension 1
# split0, split1, split2 = tf.split(value, num_or_size_splits=3, axis=1)
# print np.array(split0.shape)
# tf.shape(split0)                # [Dimension(5) Dimension(10)]

# # 2) squeeze
# value = tf.ones((1, 2, 1, 3, 1, 1), tf.int32)
# # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
# tf.shape(tf.squeeze(value))       # ==> [2, 3]
# squeeze = tf.squeeze(value)
# print np.array(tf.squeeze(value).shape)
# # shape(squeeze(t, [2, 4]))       # ==> [1, 2, 3, 1]
# with tf.Session() as sess:
#     result = sess.run([squeeze])
#     print result[0].shape
#     print result[0]

# # 3) one_hot
# x = tf.concat([tf.ones((10,), dtype=tf.int32), tf.zeros((5,), dtype=tf.int32)], axis=0)
# x_one_hot = tf.one_hot(x, 2)
# with tf.Session() as sess:
#     result = sess.run([x_one_hot])
#     print result[0].shape
#     print result[0]
# # rnn_inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, num_steps, x_one_hot)]

# 4) MSE