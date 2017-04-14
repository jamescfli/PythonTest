import os
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Hide messy TensorFlow warnings
warnings.filterwarnings("ignore")           # Hide messy Numpy warnings


def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'rb').read()
    data = f.decode().split('\n')           # e.g. .decode('utf8')

    sequence_length = seq_len + 1   # len 50 + 1 (for prediction)
    result = []
    for index in range(len(data) - sequence_length):    # slide stride = 1, sequence is largely overlapped
        result.append(data[index: index + sequence_length])     # len = 51
    # .. DL likes correlation in data, e.g. data augmentation

    if normalise_window:
        result = normalise_windows(result)  # (4121,51), all starts with 0.0 after normalization

    result = np.array(result)

    row = round(0.9 * result.shape[0])  # 3709
    train = result[:int(row), :]
    np.random.shuffle(train)            # shuffle for training, not testing
    x_train = train[:, :-1]             # (3709, 50)
    y_train = train[:, -1]              # (3709, )
    x_test = result[int(row):, :-1]     # (412, 50)
    y_test = result[int(row):, -1]      # (412, )

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))      # x input in RNN is one dim
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test]


def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]   # normalized to the starting point
        normalised_data.append(normalised_window)
    return normalised_data


def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model


def predict_point_by_point(model, data):
    # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


def predict_sequence_full(model, data, window_size):
    # Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted


def predict_sequences_multiple(model, data, window_size, prediction_len):
    # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]     # cut 0 at beginning
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs      # shape (8, 50)
