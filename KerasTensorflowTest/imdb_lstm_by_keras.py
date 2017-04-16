from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

# largest integer (i.e. word index) in the input should be no larger than 1999 (vocabulary size)
max_features = 20000
maxlen = 80     # words discarded after that
batch_size = 32

print('Loading data...')
# where
#   len(x_train[0]) = 218, len(x_train[0]) = 189
#   label y = 0/1, like or unlike
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
# shorten seq to np array with len = 80
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout_U=0.2, dropout_W=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
# Epoch 15/15
# 25000/25000 [==============================] - 322s - loss: 0.0113 - acc: 0.9962 - val_loss: 1.1053 - val_acc: 0.8070
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
# Test score: 1.10533880079
# Test accuracy: 0.80704
