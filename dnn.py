import numpy as np

from keras.layers import Dense
from keras.models import Sequential, load_model

import pandas as pd

import os


def build_network(ninputs, noutputs, sizes, act='relu', init='uniform', bias=True, output_activation='softmax',
                  regression=False):
    if regression: output_activation = 'relu'
    input_layer = [Dense(sizes[0], input_dim=ninputs, activation=act, kernel_initializer=init, use_bias=bias)]
    output_layer = [Dense(noutputs, activation=output_activation, kernel_initializer=init, use_bias=bias)]
    hidden_layers = [Dense(size, activation=act, kernel_initializer=init, use_bias=bias) for size in sizes[1:]]
    dnn = Sequential(input_layer + hidden_layers + output_layer)
    dnn.compile(loss=('categorical_crossentropy' if not regression else 'mape'), optimizer='adam',
                metrics=(['acc'] if not regression else []))
    dnn.summary()
    return dnn


def load_train_data(fn, num_ys=4):
    data = pd.read_csv(fn, sep=';', header=0, index_col=0)
    ncols = len(data.columns)
    return data.iloc[:, :ncols - num_ys], data.iloc[:, ncols - num_ys:]


def extract_split_from_prediction_data(pred_fn, data_fn, xs, ys):
    def extract_instances(fn, skip_first):
        with open(fn, 'r') as fp:
            return [line.split(';')[0] for line in fp.readlines()[1 if skip_first else 0:] if ';' in line]

    validation_instances = extract_instances(pred_fn, skip_first=False)
    ordered_instances = extract_instances(data_fn, skip_first=True)
    training_instances = [ instance for instance in ordered_instances if instance not in validation_instances]

    def only_training(src_set, inverse):
        return src_set.drop(validation_instances if not inverse else training_instances)


    train_xs = only_training(xs, inverse=False)
    train_ys = only_training(ys, inverse=False)
    val_xs = only_training(xs, inverse=True)
    val_ys = only_training(ys, inverse=True)

    return (train_xs, train_ys), (val_xs, val_ys)


if __name__ == '__main__':
    path_prefix = 'combined_j30_upto_120_rgen/'
    num_classes = 2
    skip_model_load = True

    np.random.seed(0)
    xs, ys = load_train_data(path_prefix + 'whoisbetter.csv', num_classes)

    train, validation = extract_split_from_prediction_data(path_prefix + 'predictions.csv', path_prefix + 'whoisbetter.csv', xs, ys)
    train_xs, train_ys = train
    val_xs, val_ys = validation

    model_fn = 'trained_model.h5'

    if os.path.isfile(model_fn) and not skip_model_load:
        dnn = load_model(model_fn)
    else:
        dnn = build_network(ninputs=len(xs.columns), noutputs=len(ys.columns), sizes=[190, 64, 32, 16],
                            regression=False)
        dnn.fit(train_xs, train_ys, validation_data=(val_xs, val_ys), batch_size=10, epochs=1000, verbose=2,
                shuffle=True)
        dnn.save(model_fn)

    res = dnn.predict(val_xs)
    best_actual = np.argmax(val_ys.values, axis=1)
    best_pred = np.argmax(res, axis=1)
    diff = np.subtract(best_actual, best_pred)

    print(best_pred)
    print(best_actual)
    print(diff)
    print(f'accuracy: {1-np.sum(np.abs(diff)) / len(diff)}')
