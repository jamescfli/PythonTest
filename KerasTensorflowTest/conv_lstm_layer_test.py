# use of a convolutional LSTM network

from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pylab as plt     # pylab is a module in matplotlib that gets installed alongside matplotlib


seq = Sequential()
# input_shape parameter of the first layer expects the shape without the batch_size
# None in input_shape=(None, 40, 40, 1) corresponds to the n_frames, or time steps.
# The actual input shape is (None, None, 40, 40, 1)
seq.add(ConvLSTM2D(filters=40,
                   kernel_size=(3, 3),
                   input_shape=(None, 40, 40, 1),   # nb_channels = 1
                   padding='same',
                   return_sequences=True))          # many to many
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(filters=40,
                   kernel_size=(3,3),
                   padding='same',
                   return_sequences=True))
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(filters=40,
                   kernel_size=(3,3),
                   padding='same',
                   return_sequences=True))
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(filters=40,
                   kernel_size=(3,3),
                   padding='same',
                   return_sequences=True))
seq.add(BatchNormalization())
seq.add(Conv3D(filters=1,                           # binary output
               kernel_size=(3,3,3),
               activation='sigmoid',
               padding='same',
               data_format='channels_last'))    # channel is the last dim of the output tensor
seq.compile(loss='binary_crossentropy', optimizer='adadelta')

# create movies with bigger size (80x80) and then select a 40x40 window
def generate_movies(n_samples=1200, n_frames=15):
    row = 80
    col = 80
    noisy_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)
    shifted_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)

    for i in range(n_samples):
        # 3 to 7 moving squaures
        n = np.random.randint(3, 8)

        for j in range(n):
            xstart = np.random.randint(20, 60)
            ystart = np.random.randint(20, 60)
            # direction of motion
            directionx = np.random.randint(0, 3) - 1    # -1, 0, +1
            directiony = np.random.randint(0, 3) - 1
            # size of square
            w = np.random.randint(2, 4)

            for t in range(n_frames):
                x_shift = xstart + directionx * t
                y_shift = ystart + directiony * t
                noisy_movies[i, t, x_shift - w : x_shift + w, y_shift - w : y_shift + w, 0] += 1

                # to train the network to be robust and still consider it as a pixel belonging to a square
                if np.random.randint(0, 2):
                    noise_f = (-1)**np.random.randint(0, 2)
                    noisy_movies[i, t,
                                 x_shift - w - 1 : x_shift + w + 1,
                                 y_shift - w - 1 : y_shift + w + 1,
                                 0] += noise_f * 0.1
                    x_shift = xstart + directionx * (t+1)
                    y_shift = ystart + directiony * (t+1)
                    shifted_movies[i, t, x_shift - w: x_shift + w, y_shift - w: y_shift + w, 0] += 1

    # cut to 40x40
    noisy_movies = noisy_movies[::, ::, 20:60, 20:60, ::]
    shifted_movies = shifted_movies[::, ::, 20:60, 20:60, ::]
    noisy_movies[noisy_movies >= 1] = 1
    shifted_movies[shifted_movies >= 1] = 1
    return noisy_movies, shifted_movies

# train
noisey_movies, shifted_movies = generate_movies(n_samples=1200)
seq.fit(noisey_movies[:1000], shifted_movies[:1000], batch_size=10, epochs=300, validation_split=0.05)
# .. 3170s for one epoch on MBP

# test: feed with first 7 positions and predict the new positions
which = 1004
track = noisey_movies[which][:7, ::, ::, ::]

for j in range(16):
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)
# compare with ground truth
track2 = noisey_movies[which][::, ::, ::, ::]
for i in range(15):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)

    if i >= 7:
        ax.text(1, 3, 'Prediction', fontsize=20, color='w')
    else:
        ax.text(1, 3, 'Initial trajectory', fontsize=20)

    toplot = track[i, ::, ::, 0]

    plt.imshow(toplot)
    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth', fontsize=20)

    toplot = track2[i, ::, ::, 0]
    if i >= 2:
        toplot = shifted_movies[which][i-1, ::, ::, 0]

    plt.imshow(toplot)
    plt.savefig('%i_animate.png' % (i+1))
