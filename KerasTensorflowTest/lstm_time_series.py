import time
import lstm_class.lstm as lstm
import matplotlib.pyplot as plt


def plot_results(predicted_data, true_date):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_date, label='Ground Truth')
    ax.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='Ground Truth')
    for i, data in enumerate(predicted_data):
        padding = [None for p in xrange(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


if __name__ == '__main__':
    global_start_time = time.time()
    epochs = 1
    seq_len = 50

    print '> loading data...'

    X_train, y_train, X_test, y_test = lstm.load_data('lstm_class/sp500.csv', seq_len, True)    # with 4171 lines

    print '> Data loaded. Compile...'

    model = lstm.build_model([1, 50, 100, 1])

    # Train on 3523 samples, validate on 186 samples
    model.fit(X_train, y_train, batch_size=512, nb_epoch=epochs, validation_split=0.05)

    predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)

    print 'Training duration:', time.time()-global_start_time
    plot_results_multiple(predictions, y_test, 50)
