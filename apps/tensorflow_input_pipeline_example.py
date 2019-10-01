import argparse
# import tensorflow as tf
import pandas as pd


def main(dataset_file: str):
    data_set_raw = pd.read_hdf(dataset_file, key='df')
    print(data_set_raw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', required=True, type=str)
    cmd_args = parser.parse_args()
    main(cmd_args.dataset_file)
