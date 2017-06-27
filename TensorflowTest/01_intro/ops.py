import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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

# 5) ELU
x_np = np.linspace(-5, 5, 50)
x_t = tf.constant(x_np, dtype=tf.float32)
y_elu_t = tf.nn.elu(x_t, name='elu')

alpha = 1.6732632423543772848170429916717
scale = 1.0507009873554804934193349852946
# # SELU = ELU when
# alpha = 1
# scale = 1
y_selu_t = scale*tf.where(x_t>=0.0, x_t, alpha*tf.nn.elu(x_t))

with tf.Session() as sess:
    y_elu_np = sess.run(y_elu_t)
    y_selu_np = sess.run(y_selu_t)
    plt.plot(x_np, y_elu_np, '-b', label='elu')
    plt.plot(x_np, y_selu_np, '-.r', label='selu')
    plt.legend()
    plt.show()