import argparse
import tensorflow as tf
import pandas as pd
import numpy as np

from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

data_len = 10


def build_model(rate):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[data_len]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(rate)
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


def plot_history(history, training_size, rate):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'], label='Val Error')
    plt.legend()
    plt.title("training_size = " + str(training_size) + ",rate = " + str(rate))

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label='Val Error')
    plt.legend()
    plt.title("training_size = " + str(training_size) + ",rate = " + str(rate))
    plt.savefig("../Informe2/figs/performances/feedforward-MSE-training_size_" + str(training_size) + "rate_" + str(rate) + ".pdf")
    #plt.show()


def main(dataset_file: str):
    data_set_raw = pd.read_hdf(dataset_file, key='df')
    data_set_raw = data_set_raw.map(clean_data_used)
    dataset_x = data_set_raw.map(lambda x: x[0]).dropna()
    dataset_y = data_set_raw.map(lambda x: x[1]).dropna()

    if (dataset_y.shape[0] != dataset_x.shape[0]) or not (dataset_x.index == dataset_y.index).all():
        print("error: feature and labels with different larges")
        exit(1)
    training_size = [0.6, 0.7, 0.8, 0.85]
    rates = [0.1, 0.01, 0.001, 0.0001]

    for size in training_size:
        msk = np.random.rand(len(dataset_x)) < size
        train_x = dataset_x[msk]
        validation_x = dataset_x[~msk]

        train_y = dataset_y[msk]
        validation_y = dataset_y[~msk]

        train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        validation_data = tf.data.Dataset.from_tensor_slices((validation_x, validation_y))
        for rate in rates:
            model = build_model(rate)

            logdir = "tensorboards_logs_first_try/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

            history = model.fit(train_data, validation_data=validation_data,  epochs=200, verbose=0, callbacks=[tensorboard_callback])
            plot_history(history, size, rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str, default="../datasets/iron_dataset-type.h5")
    cmd_args = parser.parse_args()
    main(cmd_args.dataset_file)
