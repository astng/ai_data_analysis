import argparse
import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers


data_len = 10


def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[data_len - 1]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model


def clean_data_used(data):
    global data_len
    if len(data) < data_len + 1:
        return np.nan, np.nan
    else:
        if pd.isna(data[0:data_len+1]).any():
            return np.nan, np.nan
        return data[0:data_len], data[data_len]


def decode_arrays(encoded_array):
    return np.frombuffer(encoded_array.numpy(), dtype=np.float32)


def decode_arrays_tf(tf_encoded):
    [decode, ] = tf.py_function(decode_arrays, [tf_encoded], [np.float32])
    return decode


def main(dataset_file: str):
    data_set_raw = pd.read_hdf(dataset_file, key='df')
    data_set_raw = data_set_raw.map(clean_data_used)
    dataset_x = data_set_raw.map(lambda x: x[0]).dropna()
    dataset_y = data_set_raw.map(lambda x: x[1]).dropna()

    if (dataset_y.shape[0] != dataset_x.shape[0]) or not (dataset_x.index == dataset_y.index).all():
        print("error: feature and labels with different larges")
        exit(1)

    dataset_cleaned_pandas = dataset_x.map(lambda x: x[~np.isnan(x)].astype(np.float32).tobytes())
    dataset_tf = tf.data.Dataset.from_tensor_slices(dataset_cleaned_pandas)
    dataset_tf = dataset_tf.map(decode_arrays_tf)
    dataset_tf_y = tf.data.Dataset.from_tensor_slices(dataset_y)

    for features, output_desire in zip(dataset_tf.take(100), dataset_tf_y.take(100)):
        print('Features: ', features, 'desired_output', output_desire)
        print(features.shape, output_desire.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', required=True, type=str)
    cmd_args = parser.parse_args()
    main(cmd_args.dataset_file)
