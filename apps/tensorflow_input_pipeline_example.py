import argparse
import tensorflow as tf
import pandas as pd
import numpy as np


def decode_arrays(encoded_array):
    return np.frombuffer(encoded_array.numpy(), dtype=np.float32)


def decode_arrays_tf(tf_encoded):
    [decode, ] = tf.py_function(decode_arrays, [tf_encoded], [np.float32])
    return decode


def main(dataset_file: str):
    data_set_raw = pd.read_hdf(dataset_file, key='df')
    dataset_cleaned_pandas = data_set_raw.map(lambda x: x[~np.isnan(x)].astype(np.float32).tobytes())
    dataset_tf = tf.data.Dataset.from_tensor_slices(dataset_cleaned_pandas)
    dataset_tf = dataset_tf.map(decode_arrays_tf)

    for features in dataset_tf.take(100):
        print('Features: ', features)
        print(features.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', required=True, type=str)
    cmd_args = parser.parse_args()
    main(cmd_args.dataset_file)
