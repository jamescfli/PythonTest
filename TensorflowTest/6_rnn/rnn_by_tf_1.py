import numpy as np


print("Expected cross entropy loss if the model:")
print("- learns neither dependency:", -(0.625 * np.log(0.625) +
                                      0.375 * np.log(0.375)))
print("- learns first dependency:  ",
      -0.5 * (0.875 * np.log(0.875) + 0.125 * np.log(0.125))
      -0.5 * (0.625 * np.log(0.625) + 0.375 * np.log(0.375)))
print("- learns both dependencies: ",
      -0.50 * (0.75 * np.log(0.75) + 0.25 * np.log(0.25))
      - 0.25 * (2 * 0.50 * np.log (0.50)) - 0.25 * (0))

# Expected cross entropy loss if the model:
# ('- learns neither dependency:', 0.66156323815798213)
# ('- learns first dependency:  ', 0.51916669970720941)
# ('- learns both dependencies: ', 0.4544543674493905)

import tensorflow as tf
import matplotlib.pyplot as plt

num_steps = 5
batch_size = 200
num_classes = 2
state_size = 4
learning_rate = 0.1


def gen_data(size=1000000):
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i-3] == 1:
            threshold += 0.5
        if X[i-8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)


def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length*i : batch_partition_length(i+1)]
        data_y[i] = raw_y[batch_partition_length*i : batch_partition_length(i+1)]
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i*num_steps : (i+1)*num_steps]
        y = data_y[:, i*num_steps : (i+1)*num_steps]
        yield (x, y)


def gen_epochs(n, num_steps):
    for i in range(n):
        yield gen_batch(gen_data(), batch_size, num_steps)
