import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler


def reducer(epoch):
    # increase epoch by 1 as epochs are counted from 0
    epoch = epoch + 1
    if epoch < 5:
        return 0.001
    else:
        # the return of the scheduler must be a float
        return 0.001 * np.exp(0.1 * (5 - epoch))


def train_test_split(df, size=0.25):
    split_row = len(df) - int(size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data


def build_model(input_data, output_size, neurons=64, drop=0.25):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2]), return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(neurons, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(neurons)),
        tf.keras.layers.Dropout(drop),
        tf.keras.layers.Dense(neurons, activation='relu'),
        tf.keras.layers.Dropout(drop),
        tf.keras.layers.Dense(units=output_size, activation='relu')
    ])
    #optimizer = tf.keras.optimizers.RMSprop(0.001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse', 'mape'])
    return model


def extract_window_data(df, window=10):
    window_data = []
    for idx in range(len(df) - window):
        tmp = df[idx: (idx + window)].copy()
        window_data.append(tmp.values)
    return np.array(window_data)


def prepare_data(df, target_col, window=10, testing_size=0.25):
    train_data, test_data = train_test_split(df, testing_size)
    x_train = extract_window_data(train_data, window)
    x_test = extract_window_data(test_data, window)

    y_train = train_data[target_col][window:].values
    y_test = test_data[target_col][window:].values

    return train_data, test_data, x_train, x_test, y_train, y_test


def plot_history(history, essay):
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='all')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Abs Error [%PPM]')
    ax2.set_ylabel('Mean Square Error')
    ax3.set_ylabel('Mean Abs Perc Error')
    fig.suptitle("Bidirectional 3xLSTM with all data input as numeric values")
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    ax1.plot(hist['epoch'], hist['mae'], label='Training Error')
    ax1.plot(hist['epoch'], hist['val_mae'], label='Validation Error')
    ax2.plot(hist['epoch'], hist['mse'], label='Training Error')
    ax2.plot(hist['epoch'], hist['val_mse'], label='Validation Error')
    ax3.plot(hist['epoch'], hist['mape'], label='Training Error')
    ax3.plot(hist['epoch'], hist['val_mape'], label='Validation Error')
    ax1.legend()
    ax1.grid(True)
    ax2.legend()
    ax2.grid(True)
    ax3.legend()
    ax3.grid(True)
#    plt.show(block=False)
    plt.savefig("../figures/essays_predictions/" + essay + "_3xbidir_lstm_errors.pdf")

def plot_predictions(network_model, df, testing_set, essay):
    global batch_size
    component_results = df[essay][df['component'] == comp].reset_index()
    predicted_changes = df["change"][df['component'] == comp].reset_index()
    plt.figure()
    plt.plot(component_results[essay], 'b', label='True values')
    predicted_changes = predicted_changes.change*component_results[essay]
    plt.plot(predicted_changes[predicted_changes != 0].index, predicted_changes[predicted_changes != 0].values, 'g<', label='_nolegend')
    predictions = network_model.predict(testing_set, batch_size=batch_size)
#    predictions = normal_predictions * norm_std + norm_mean
    initial_zeros = np.zeros((window_len, 1))
    predictions = np.vstack([initial_zeros, predictions])
    plt.plot(predictions, 'r--', label='Predictions')
    plt.xlabel('Sample')
    plt.ylabel(essay + ' [PPM]')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.title('Bidirectional 3xLSTM prediction for comp=' + str(comp) + ', tipo_comp=' + str(tipo_comp))
#    plt.show()
    plt.savefig("../figures/essays_predictions/" + essay + "_3xbidir_lstm_prediction.pdf")


np.random.seed(420)  # from numpy
tf.random.set_seed(420)  # from tensorflow

window_len = 10
test_size = 0.25
normalise = True
lstm_neurons = 64
epochs = 1
batch_size = 8
dropout = 0.5
comp = 3752
tipo_comp = 682
iron_denormalise = {}


def main(essay: str, dataset_file: str):
    print(essay)
    if essay != "iron":
        dataset_file = dataset_file.replace("iron", essay)
    data_set_raw = pd.read_hdf(dataset_file, key='df')
    data_set_raw = pd.DataFrame(data_set_raw.values.tolist())
    data_set_raw.columns = ["component", "component_type", "machine_type", "change", essay + "LSC", essay + "LSM",
                            "h_k_lubricante", essay]
    data_set_raw["change"] = 1*data_set_raw["change"]
    if normalise:
        original_df = data_set_raw.copy()
        for col in data_set_raw.columns:
            if col == essay:
                mean, std = data_set_raw[col].mean(), data_set_raw[col].std()
            elif col == "component":
                aux = data_set_raw[data_set_raw['component'] == comp]
            if col != "change" and col != essay:
                data_set_raw[col] = (data_set_raw[col] - data_set_raw[col].mean()) / data_set_raw[col].std()
    train, test, x_train, x_test, y_train, y_test = prepare_data(data_set_raw, essay, window=window_len)
    x_test_to_predict = prepare_data(aux, essay, window=window_len, testing_size=1)[3]
    model = build_model(x_train, output_size=1, neurons=lstm_neurons, drop=dropout)
    logdir = "tensorboards_logs_lstm/scalars/whole_input-numerics"
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=False,
                        callbacks=[tensorboard_callback, LearningRateScheduler(reducer), early_stop],
                        validation_data=(x_test, y_test))
    plot_history(history, essay)
    plot_predictions(model, original_df, x_test_to_predict, essay)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--essay', type=str, default="iron")
    parser.add_argument('--dataset_file', type=str, default="../datasets/iron_dataset.h5")
    cmd_args = parser.parse_args()
    main(cmd_args.essay, cmd_args.dataset_file)