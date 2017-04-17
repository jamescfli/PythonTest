# to generate text from Nietzsche's writings
# At least 20 epochs are required
# this script on new data, make sure your corpus has at least ~100k characters. ~1M is better.

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('corpus length:', len(text))  # 600901

chars = sorted(list(set(text)))
print('total chars:', len(chars))   # 59
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
# # or
# indices_char = dict(enumerate(chars))
# vocab_to_idx = dict(zip(indices_char.values(), indices_char.keys()))

max_len = 40
step = 3    # not too much correlation for training
sentences = []
next_chars = []
for i in range(0, len(text) - max_len, step):
    sentences.append(text[i:i+max_len])
    next_chars.append(text[i+max_len])
print('nb sequences:', len(sentences))  # 200287 ~= 600901/step

print('Vectorization...')
X = np.zeros((len(sentences), max_len, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1     # one hot coding
    y[i, char_indices[next_chars[i]]] = 1

# build single LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(max_len, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # sample an index from a prob array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)      # n, pvals, size
    return np.argmax(probs)

for iteration in range(1, 60):
    print()
    print('-' * 30)
    print('Iteration', iteration)
    model.fit(X, y,
              batch_size=128,
              epochs=1)
    start_index = random.randint(0, len(text) - max_len - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:  # higher -> more avg on each probability
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index : start_index + max_len]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, max_len, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1
            preds = model.predict(x, verbose=0)[0]  # preds = size 59 probability vector
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

# .. text is making greater sense after each epoch
# .. and increasing diversity makes less sense to the script due to averaging out the probability

# the last epoch
# ------------------------------
# Iteration 59
# Epoch 1/1
# 200287/200287 [==============================] - 260s - loss: 1.3405      loss does not decrease too much
#
# ----- diversity: 0.2
# ----- Generating with seed: "at we may be standing on
# the threshold o"
# at we may be standing on
# the threshold one may be problem of the sacrifice and philosophy and species of the strange of the same and the present the subjection of man is not the strange of the man of the same man is not conception of the same intellect of the same things are subjection of the strange and philosophy and self-spiritual interest the presentience and spectations of the strange and serious and the present strange of the same
#
# ----- diversity: 0.5
# ----- Generating with seed: "at we may be standing on
# the threshold o"
# at we may be standing on
# the threshold of the best the greaters and conscious to
# self consequences of the sense of the soul of the long away self-spiritual emplete they will let them. it is just the powerful for the form of small desirements, and world and
# language with any serious of the entirely believe of the religion, have not still the philosophy and the man and man of the true. the man and conception of man as the christian high
# t
#
# ----- diversity: 1.0
# ----- Generating with seed: "at we may be standing on
# the threshold o"
# at we may be standing on
# the threshold of to him may near it ascience; the i nown that the est that attained yours obk.=--as thereaf ofition aguntice, the remailed their christering intellecter--a destacte, all nature does not conection couradists
#
# k
# 12. it of this investigates
# be endures that is not prepaited of his cysicions the human's--mited and xcipless of longancic francion to distinction ofly above this detacity to
# considerations
#
# ----- diversity: 1.2      still, does not make too much sense
# ----- Generating with seed: "at we may be standing on
# the threshold o"
# at we may be standing on
# the threshold of the
# mansifuerisationion--it can, dones, the idealise exked,
# unbutual predely them, now bying to havely.
# aslelung, whether
# extraordinary servine grals madely, thround; how woves shadders, mushtings." have to knew of after nater: how-pumbain danger, aptitude of -one would be yours, cause, perhaps use of
# towards towards this modering
# has ebever his powish in the pees natural
# myself-expect it even r
