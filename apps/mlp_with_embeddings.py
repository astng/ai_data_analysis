import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from keras import callbacks
from keras import backend as k
from keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
import matplotlib.pyplot as plt


def reducer(epoch):
    # increase epoch by 1 as epochs are counted from 0
    epoch = epoch + 1
    if epoch < 5:
        return 0.001
    else:
        # the return of the scheduler must be a float
        return 0.001 * np.exp(0.1 * (5 - epoch))


def get_keras_dataset(df):
    x = {str(col): np.array(df[col]) for col in df.columns}
    return x


def plot_history(history):
    print("here")
    fig, (ax1, ax2) = plt.subplots(2, sharex='all')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Abs Error [PPM]')
    ax2.set_ylabel('Mean Square Error')
    fig.suptitle("Embeddings+Dense layers")
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    ax1.plot(hist['epoch'], hist['mae'], label='Training Error')
    ax1.plot(hist['epoch'], hist['val_mae'], label='Validation Error')
    ax2.plot(hist['epoch'], hist['mse'], label='Training Error')
    ax2.plot(hist['epoch'], hist['val_mse'], label='Validation Error')
    ax1.legend()
    ax1.grid(True)
    ax2.legend()
    ax2.grid(True)
#    plt.show(block=False)
    plt.savefig("last_errors.pdf")


def plot_predictions(network_model, df, testing_set):
    global batch_size
    component_results = df["iron"][df['component'] == comp].reset_index()
    predicted_changes = df["change"][df['component'] == comp].reset_index()
    plt.figure()
    plt.plot(component_results["iron"], 'b', label='True values')
    predicted_changes = predicted_changes.change*component_results.iron
    plt.plot(predicted_changes[predicted_changes != 0].index, predicted_changes[predicted_changes != 0].values, 'g<',
             label='_nolegend')
    predictions = network_model.predict(testing_set, batch_size=batch_size)
    initial_zeros = np.zeros((window_len, 1))
    predictions = np.vstack([initial_zeros, predictions])
    plt.plot(predictions, 'r--', label='Predictions')
    plt.xlabel('Sample')
    plt.ylabel('Iron [PPM]')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.title('Prediction (w/embeddings) for comp=' + str(comp) + ', tipo_comp=' + str(tipo_comp))
#    plt.show()
    plt.savefig("last_prediction.pdf")


def add_low_dim_features(data, testing_size = 0.25):
    df = data.copy()
    numeric_cols = ['change', 'ironLSC', 'ironLSM', 'h_k_lubricante']
    categorical_cols = ['component', 'component_type', 'machine_type']

    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])

    pca = PCA(n_components=3)
    _X = pca.fit_transform(df[numeric_cols + categorical_cols])
    pca_data = pd.DataFrame(_X, columns=["PCA1", "PCA2", "PCA3"])
    df[["PCA1", "PCA2", "PCA3"]] = pca_data

    fica = FastICA(n_components=3)
    _X = fica.fit_transform(df[numeric_cols + categorical_cols])
    fica_data = pd.DataFrame(_X, columns=["FICA1", "FICA2", "FICA3"])
    df[["FICA1", "FICA2", "FICA3"]] = fica_data

    tsvd = TruncatedSVD(n_components=3)
    _X = tsvd.fit_transform(df[numeric_cols + categorical_cols])
    tsvd_data = pd.DataFrame(_X, columns=["TSVD1", "TSVD2", "TSVD3"])
    df[["TSVD1", "TSVD2", "TSVD3"]] = tsvd_data

    grp = GaussianRandomProjection(n_components=3)
    _X = grp.fit_transform(df[numeric_cols + categorical_cols])
    grp_data = pd.DataFrame(_X, columns=["GRP1", "GRP2", "GRP3"])
    df[["GRP1", "GRP2", "GRP3"]] = grp_data

    srp = SparseRandomProjection(n_components=3)
    _X = srp.fit_transform(df[numeric_cols + categorical_cols])
    srp_data = pd.DataFrame(_X, columns=["SRP1", "SRP2", "SRP3"])
    df[["SRP1", "SRP2", "SRP3"]] = srp_data

    numeric_cols.extend(pca_data.columns.values)
    numeric_cols.extend(fica_data.columns.values)
    numeric_cols.extend(tsvd_data.columns.values)
    numeric_cols.extend(grp_data.columns.values)
    numeric_cols.extend(srp_data.columns.values)
    train_df, test_df = train_test_split(df, test_size=testing_size, random_state=42)
    return train_df, test_df, categorical_cols, numeric_cols


np.random.seed(420)  # from numpy
window_len = 1
test_size = 0.25
neurons = 128
epochs = 200
batch_size = 8
dropout = 0.2
comp = 3752
tipo_comp = 682
data_len = 10

def main(dataset_file: str):
    data_set_raw = pd.read_hdf(dataset_file, key='df')
    data_set_raw = pd.DataFrame(data_set_raw.values.tolist())
    data_set_raw.columns = ["component", "component_type", "machine_type", "change", "ironLSC", "ironLSM",
                            "h_k_lubricante", "iron"]
    data_set_raw["change"] = 1 * data_set_raw["change"]
    original_df = data_set_raw.copy()

    train_df, test_df, categorical_cols, numeric_cols = add_low_dim_features(data_set_raw)

    k.clear_session()

    feature_cols = numeric_cols + categorical_cols
    target_col = 'iron'

    cat_inputs = []
    num_inputs = []
    embeddings = []
    embedding_layer_names = []
    emb_n = 10

    for col in categorical_cols:
        _input = layers.Input(shape=(1,), name=col)
        _embed = layers.Embedding(data_set_raw[col].max() + 1, emb_n, name=col + '_emb')(_input)
        cat_inputs.append(_input)
        embeddings.append(_embed)
        embedding_layer_names.append(col + '_emb')

    for col in numeric_cols:
        numeric_input = layers.Input(shape=(1,), name=col)
        num_inputs.append(numeric_input)

    merged_num_inputs = layers.concatenate(num_inputs)
    numeric_dense = layers.Dense(20, activation='relu')(merged_num_inputs)

    merged_inputs = layers.concatenate(embeddings)
    spatial_dropout = layers.SpatialDropout1D(dropout)(merged_inputs)
    flat_embed = layers.Flatten()(spatial_dropout)
    all_features = layers.concatenate([flat_embed, numeric_dense])

    x = layers.Dropout(dropout)(layers.Dense(100, activation='relu')(all_features))
    x = layers.Dropout(dropout)(layers.Dense(50, activation='relu')(x))
    x = layers.Dropout(dropout)(layers.Dense(25, activation='relu')(x))
    x = layers.Dropout(dropout)(layers.Dense(15, activation='relu')(x))

    output = layers.Dense(1, activation='relu')(x)
    model = models.Model(inputs=cat_inputs + num_inputs, outputs=output)

    model.compile(loss='mse', optimizer="adam", metrics=['mae', 'mse'])

    print(model.summary())

    tb_callback = callbacks.TensorBoard(
        log_dir="tensorboards_logs_lstm/scalars/whole_input",
    )

    bm_callback = callbacks.ModelCheckpoint(
        filepath="tensorboards_logs_lstm/scalars/whole_input/bestmodel.h5",
        save_best_only=True,
        save_weights_only=False
    )
    x_train = get_keras_dataset(train_df[feature_cols])
    y_train = np.asarray(train_df[target_col])
    x_test = get_keras_dataset(test_df[feature_cols])
    y_test = np.asarray(test_df[target_col])

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=30)

    _hist = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size,
                      callbacks=[LearningRateScheduler(reducer), early_stop], verbose=1,
                      shuffle=False)
    plot_history(_hist)
    aux = original_df[original_df['component'] == comp]
    aux_train, aux_test, _, _ = add_low_dim_features(aux, 1)
    x_test_to_predict = get_keras_dataset(aux_test[feature_cols])
    plot_predictions(model, original_df, x_test_to_predict)
#    model = keras.models.load_model("tensorboards_logs_lstm/scalars/whole_input/bestmodel.h5", compile=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str, default="../datasets/iron_dataset-whole.h5")
    cmd_args = parser.parse_args()
    main(cmd_args.dataset_file)
