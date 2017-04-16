from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb


max_features = 20000    # max feature common words
maxlen = 100
batch_size = 32

print('Loading data...')
# num_words: integer or None. Top most frequent words to consider.
# Any less frequent word will appear as 0 in the sequence data.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print("Pad sequences (samples x time)")
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)    # pad and cut to maxlen
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))    # word embedding
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# # embedding layer
# print(model.layers[0])
# print(model.layers[0].get_weights()[0].shape)
# print(model.layers[0].get_weights()[0][0, :])

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=[x_test, y_test])
# Epoch 4/4
# 25000/25000 [==============================] - 297s - loss: 0.0699 - acc: 0.9755 - val_loss: 0.7032 - val_acc: 0.8269
# Epoch 2/15
# 25000/25000 [==============================] - 297s - loss: 0.2221 - acc: 0.9137 - val_loss: 0.3512 - val_acc: 0.8449
# Epoch 7/15
# 25000/25000 [==============================] - 313s - loss: 0.0239 - acc: 0.9922 - val_loss: 0.7007 - val_acc: 0.8258
# .. already > 0.80704 for basic lstm with 15 epochs

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

# # embedding layer after training
# print(model.layers[0].get_weights()[0][0, :])
