import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt


def plot_history(history):

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='all')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Abs Error [PPM]')
    ax2.set_ylabel('Mean Square Error')
    ax3.set_ylabel('Mean Abs Perc Error')
    fig.suptitle("LSTM with categorical data input")
#    legends = []
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    ax1.plot(hist['epoch'], hist['mae'])
    ax1.plot(hist['epoch'], hist['val_mae'])
    ax2.plot(hist['epoch'], hist['mse'])
    ax2.plot(hist['epoch'], hist['val_mse'])
    ax3.plot(hist['epoch'], hist['mape'])
    ax3.plot(hist['epoch'], hist['val_mape'])
#    legends.append('TrainError (size:' + str(training_size) + ')')
#    legends.append('ValError (train_size:' + str(training_size) + ')')
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
#    ax1.legend(legends)
#    ax2.legend(legends)
#    ax3.legend(legends)
    plt.show()


def train_test_split(df, size=0.25):
    split_row = len(df) - int(size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data


def build_model(input_data, output_size, neurons=64, drop=0.25):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])),
        tf.keras.layers.Dropout(drop),
        tf.keras.layers.Dense(units=output_size)
    ])
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse', 'mape'])
    return model


def normalise_minmax(df):
    return (df - df.min()) / (df.max() - df.min())


def extract_window_data(df, window=10, normal=True):
    window_data = []
    for idx in range(len(df) - window):
        tmp = df[idx: (idx + window)].copy()
        if normal:
            tmp = normalise_minmax(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)


def prepare_data(df, target_col, window=10, normal=True):
    train_data, test_data = train_test_split(df)
    x_train = extract_window_data(train_data, window, normal)
    x_test = extract_window_data(test_data, window, normal)

    y_train = train_data[target_col][window:].values
    y_test = test_data[target_col][window:].values
    if normalise:
        y_train = y_train / train_data[target_col][:-window].values - 1
        y_test = y_test / test_data[target_col][:-window].values - 1

    return train_data, test_data, x_train, x_test, y_train, y_test


np.random.seed(42)

window_len = 10
test_size = 0.25
normalise = True

lstm_neurons = 64
epochs = 50
batch_size = 4
dropout = 0.25


def main(dataset_file: str):
    data_set_raw = pd.read_hdf(dataset_file, key='df')
    data_set_raw = pd.DataFrame(data_set_raw.values.tolist())
    data_set_raw.columns = ["component", "component_type", "machine_type", "ironLSC", "ironLSM", "h_k_lubricante",
                            "iron"]
    train, test, x_train, x_test, y_train, y_test = prepare_data(data_set_raw, "iron", window=window_len,
                                                                 normal=normalise)

    model = build_model(x_train, output_size=1, neurons=lstm_neurons, drop=dropout)
    logdir = "tensorboards_logs_lstm/scalars/whole_input-numerics"
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True,
                        callbacks=[tensorboard_callback])

    plot_history(history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str, default="../datasets/iron_dataset-whole.h5")
    cmd_args = parser.parse_args()
    main(cmd_args.dataset_file)