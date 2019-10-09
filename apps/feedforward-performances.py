import argparse
import tensorflow as tf
import pandas as pd
import numpy as np

from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

data_len = 10


def build_model(neurons, rate, method):
    model = keras.Sequential([
        layers.Dense(neurons, activation='relu', input_shape=[data_len]),
        layers.Dense(neurons, activation='relu'),
        layers.Dense(1)
    ])
    if method == "RMSprop":
        optimizer = tf.keras.optimizers.RMSprop(rate)
    elif method == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=rate)
    elif method == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=rate)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model


def clean_data_used(data):
    global data_len
    if len(data) < data_len + 1:
        return np.nan, np.nan
    else:
        if pd.isna(data[0:data_len+1]).any():
            return np.nan, np.nan
        return [data[0:data_len].astype(np.float32)], np.array([data[data_len]]).astype(np.float32)


def plot_history(histories, rate, units, optimizer):
    sizes = list(histories.keys())
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Abs Error [MPG]')
    ax2.set_ylabel('Mean Square Error [$MPG^2$]')
    fig.suptitle("Opt.:" + optimizer + "-rate:" + str(rate) + "-" + str(units) + "neurons")
    legends = []
    for training_size in sizes:
        history = histories[training_size]
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        ax1.plot(hist['epoch'], hist['mae'])
        ax1.plot(hist['epoch'], hist['val_mae'])
        ax2.plot(hist['epoch'], hist['mse'])
        ax2.plot(hist['epoch'], hist['val_mse'])
        legends.append('TrainError (size:' + str(training_size) + ')')
        legends.append('ValError (train_size:' + str(training_size) + ')')
    ax1.grid(True)
    ax2.grid(True)
    ax1.legend(legends)
    ax2.legend(legends)

    plt.savefig("../Informe2/figs/performances/Opt_" + optimizer + "-rate_" + str(rate) + "-" + str(units) + "neurons" + ".pdf")
    #plt.show()


def main(dataset_file: str):
    data_set_raw = pd.read_hdf(dataset_file, key='df')
    data_set_raw = data_set_raw.map(clean_data_used)
    dataset_x = data_set_raw.map(lambda x: x[0]).dropna()
    dataset_y = data_set_raw.map(lambda x: x[1]).dropna()

    if (dataset_y.shape[0] != dataset_x.shape[0]) or not (dataset_x.index == dataset_y.index).all():
        print("error: feature and labels with different larges")
        exit(1)
    training_size = [0.7, 0.75, 0.8]
    rates = [0.01, 0.001, 0.0001]
    neurons = [32, 64, 128, 256]
    optimizers = ["Adam", "SGD", 'RMSprop']

    for optimizer in optimizers:
        for units in neurons:
            for rate in rates:
                histories = {}
                for size in training_size:
                    msk = np.random.rand(len(dataset_x)) < size
                    train_x = dataset_x[msk]
                    validation_x = dataset_x[~msk]

                    train_y = dataset_y[msk]
                    validation_y = dataset_y[~msk]

                    train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y))
                    validation_data = tf.data.Dataset.from_tensor_slices((validation_x, validation_y))

                    model = build_model(units, rate, optimizer)

                    logdir = "tensorboards_logs_first_try/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
                    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

                    history = model.fit(train_data, validation_data=validation_data,  epochs=150, verbose=0, callbacks=[tensorboard_callback])
                    histories[size] = history
                plot_history(histories, rate, units, optimizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str, default="../datasets/iron_dataset-type.h5")
    cmd_args = parser.parse_args()
    main(cmd_args.dataset_file)
