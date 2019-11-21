from pymongo import MongoClient
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

np.random.seed(420)
time_horizon = 100
data_len = int(0.1*time_horizon)
TRAIN_SPLIT = int(0.75*time_horizon)  # 75% of iron values will be used as training set
BATCH_SIZE = 3*data_len
BUFFER_SIZE = time_horizon

def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)


def create_time_steps(length):
  time_steps = []
  for i in range(-length, 0, 1):
    time_steps.append(i)
  return time_steps

def baseline(history):
  return np.mean(history)

def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
#        plt.plot(future, plot_data[i], marker[i], markersize=10)
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
#      plt.plot(time_steps, plot_data[i].flatten(), marker[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  return plt

def multi_step_plot(history, true_future, prediction):
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, 1]), label='History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.show()


def main(dataset_file: str):
#    data_set_raw = pd.read_hdf(dataset_file, key='df')
#    iron_values = data_set_raw.map(lambda x: x[-1]).dropna()
    id_component = 1037
    client = MongoClient()
    db_mongo = client['astng_stats']
    table_mongo = db_mongo['whours_data']
    query_fetch = table_mongo.find()
    all_data = pd.DataFrame(query_fetch)
    grouped_by_component = all_data.groupby(by='component')
    component_group = grouped_by_component.groups[id_component]
    component_results = all_data.iloc[component_group]["iron"].dropna().reset_index(drop=True)
    #plt.plot(component_results[:100])
    #legend.append("id_component=" + str(id_component))
    #plt.legend(legend)
    #plt.xlabel("muestras")
    #plt.ylabel("desgaste [ppm]")
    #plt.grid(True)
    #plt.show()
    dataset = component_results.values
    uni_data = dataset[:time_horizon]
    uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
    uni_train_std = uni_data[:TRAIN_SPLIT].std()
    uni_data = (uni_data-uni_train_mean)/uni_train_std
    univariate_past_history = data_len
    univariate_future_target = 0

    x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT, univariate_past_history,
                                               univariate_future_target)
    x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None, univariate_past_history,
                                           univariate_future_target)

    train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
    train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
    val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

    simple_lstm_model = tf.keras.models.Sequential([tf.keras.layers.LSTM(64, input_shape=x_train_uni.shape[-2:]),
                                                    tf.keras.layers.Dense(1)])
    simple_lstm_model.compile(optimizer='adam', loss='mae')

    for x, y in val_univariate.take(1):
        print(simple_lstm_model.predict(x).shape)


    EVALUATION_INTERVAL = time_horizon
    EPOCHS = 70

    simple_lstm_model.fit(train_univariate, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL,
                          validation_data=val_univariate, validation_steps=int(0.5*time_horizon))

    for x, y in val_univariate.take(1):
        plot = show_plot([x[0].numpy(), y[0].numpy(), simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
        plot.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str, default="../datasets/iron_dataset-whole.h5")
    cmd_args = parser.parse_args()
    main(cmd_args.dataset_file)