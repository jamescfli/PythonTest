# https://keras.io/layers/recurrent/
# stateful: Boolean (default False). If True, the last state for each sample
# at index i in a batch will be used as initial state for the sample of
# index i in the following batch.

# use stateful RNNs to model long sequences efficiently
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM

# since stateful is used, the rnn steps can be 1
tsteps = 1
batch_size = 25
epochs = 25
# number of elements ahead that are used to make the prediction
lahead = 1


def gen_cosine_amp(amp=100, period=1000, x0=0, xn=50000, step=1, k=0.0001):
    cos = np.zeros(((xn-x0) * step, 1, 1))
    for i in range(len(cos)):
        idx = x0+i*step
        cos[i, 0, 0] = amp*np.cos(2*np.pi*idx / period)
        cos[i, 0, 0] = cos[i, 0, 0] * np.exp(-k*idx)
    return cos

print('Generating data...')
cos = gen_cosine_amp()
print('Input shape:', cos.shape)

expected_output = np.zeros((len(cos), 1))
for i in range(len(cos) - lahead):
    expected_output[i, 0] = np.mean(cos[i+1 : i+lahead+1])

print('Output shape:', expected_output.shape)

print('Creating model...')
model = Sequential()
model.add(LSTM(50,
               input_shape=(tsteps, 1),
               batch_size=batch_size,
               return_sequences=True,
               stateful=True))
model.add(LSTM(50,
               return_sequences=False,
               stateful=True))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop')

print('Train...')
for i in range(epochs):
    print('Epoch', i, '/', epochs)
    # the last state for sample i in a batch will be used as initial state for sample i in the next batch
    model.fit(cos, expected_output,
              batch_size=batch_size,
              epochs=1,
              verbose=1,
              shuffle=False)
    model.reset_states()

print('Predicting')
predicted_output = model.predict(cos, batch_size=batch_size)

print('Plotting')
plt.subplot(211)
plt.plot(expected_output)
plt.title('Expected')
plt.subplot(212)
plt.plot(predicted_output)
plt.title('Predicted')
plt.show()
