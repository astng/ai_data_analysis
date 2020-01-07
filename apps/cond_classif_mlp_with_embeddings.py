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
import seaborn as sns


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


def condition_color(lsm, lsc, value):
    if value >= lsc:
        return "rojo"
    elif value >= lsm:
        return "amarillo"
    return "verde"

def plot_condition_classification(network_model, x_test, y_test, essay):
    global batch_size
    conditions = ["verde", "amarillo", "rojo"]
    prediction_values = network_model.predict(x_test, batch_size=batch_size)
    predictions = {"verde": [0, 0, 0], "amarillo": [0, 0, 0], "rojo": [0, 0, 0]}
    for t in range(len(y_test)):
        predicted_condition = condition_color(x_test[essay + "LSM"][t], x_test[essay + "LSC"][t], prediction_values[t])
        real_condition = condition_color(x_test[essay + "LSM"][t], x_test[essay + "LSC"][t], y_test[t])
        if real_condition == "rojo":
            predictions[predicted_condition][2] += 1
        elif real_condition == "amarillo":
            predictions[predicted_condition][1] += 1
        elif real_condition == "verde":
            predictions[predicted_condition][0] += 1
    print(predictions)
    accuracy = (predictions["rojo"][2] + predictions["amarillo"][1] + predictions["verde"][0]) / (
                sum(predictions["rojo"]) + sum(predictions["amarillo"]) + sum(predictions["verde"]))
    cm = []
    for condition in conditions:
        cm.append(predictions[condition])
    df_cm = pd.DataFrame(cm, index=conditions, columns=conditions)
    ax = sns.heatmap(df_cm, vmax=1, center=0, cmap='coolwarm', square=True, annot=True, fmt="d")
    ax.set_title('Accuracy: ' + str(round(100*accuracy, 2)) + "%")
    plt.xlabel("Real condition")
    plt.ylabel("Predicted condition")
#    plt.show()
    plt.savefig("figures/condition_classification/" + essay + "_mlp_with_embeddings.pdf")


def add_low_dim_features(data, essay, testing_size = 0.25):
    df = data.copy()
    numeric_cols = ['change', essay + 'LSC', essay + 'LSM', 'h_k_lubricante']
    categorical_cols = ['component', 'component_type', 'machine_type']

    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

#    df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])

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
tf.random.set_seed(420)
window_len = 1
test_size = 0.25
neurons = 128
epochs = 100
batch_size = 8
dropout = 0.2
comp = 3752
tipo_comp = 682
data_len = 10

def main(essay: str, dataset_file: str):
    print(essay)
    if essay != "iron":
        dataset_file = dataset_file.replace("iron", essay)
    data_set_raw = pd.read_hdf(dataset_file, key='df')
    data_set_raw = pd.DataFrame(data_set_raw.values.tolist())
    data_set_raw.columns = ["component", "component_type", "machine_type", "change", essay + "LSC", essay + "LSM",
                            "h_k_lubricante", essay]
    data_set_raw["change"] = 1 * data_set_raw["change"]
    original_df = data_set_raw.copy()

    train_df, test_df, categorical_cols, numeric_cols = add_low_dim_features(data_set_raw, essay)

    k.clear_session()

    feature_cols = numeric_cols + categorical_cols
    target_col = essay

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

    x_train = get_keras_dataset(train_df[feature_cols])
    y_train = np.asarray(train_df[target_col])
    x_test = get_keras_dataset(test_df[feature_cols])
    y_test = np.asarray(test_df[target_col])
    bm_callback = callbacks.ModelCheckpoint(
        filepath="models/condition_classification/" + essay + "_best_model.h5",
        save_best_only=True,
        save_weights_only=False
    )
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20)

    _hist = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size,
                      callbacks=[LearningRateScheduler(reducer), early_stop, bm_callback], verbose=1,
                      shuffle=False)
#    plot_history(_hist)
    plot_condition_classification(model, x_test, y_test, essay)

#    model = keras.models.load_model("tensorboards_logs_lstm/scalars/whole_input/bestmodel.h5", compile=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--essay', type=str, default="iron")
    parser.add_argument('--dataset_file', type=str, default="datasets/iron_dataset.h5")
    cmd_args = parser.parse_args()
    main(cmd_args.essay, cmd_args.dataset_file)
