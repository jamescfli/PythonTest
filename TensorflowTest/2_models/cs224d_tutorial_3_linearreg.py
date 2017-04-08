from __future__ import print_function
import numpy as np
# import seaborn
import matplotlib.pyplot as plt
import tensorflow as tf

# define input
X_data = np.arange(100, step=.1)
y_data = X_data + 20 * np.sin(X_data/10)
# print X_data

plt.scatter(X_data, y_data)
# plt.show()

n_samples = 1000
batch_size = 100

X_data = np.reshape(X_data, (n_samples, 1))
y_data = np.reshape(y_data, (n_samples, 1))

X = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))

with tf.variable_scope("linearreg"):
    W = tf.get_variable("weights", (1,1),
                        initializer=tf.random_normal_initializer())
    b = tf.get_variable("bias", (1,),
                        initializer=tf.constant_initializer(0.0))
    y_pred = tf.add(tf.matmul(X, W), b)
    loss = tf.reduce_sum((y - y_pred)**2/n_samples)

optimizer = tf.train.AdamOptimizer()
# TensorFlow scope is not python scope! Python variable loss is still visible
opt_operation = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # sess.run([opt_operation], feed_dict={X: X_data, y: y_data})
    for _ in range(5000):
        indices = np.random.choice(n_samples, batch_size)
        X_batch, y_batch = X_data[indices], y_data[indices]
        _, loss_val, w_val, b_val = sess.run([opt_operation, loss, W, b], feed_dict={X:X_batch, y:y_batch})
        # print('loss:', loss_val, 'W:', w_val, 'b:', b_val)

linear_regression = X_data*w_val + b_val
plt.plot(X_data, linear_regression, '-r')
plt.show()