import numpy as np

from keras.layers import Dense
from keras.models import Sequential, load_model

import pandas as pd

import os


def build_network(ninputs, noutputs, sizes, act='relu', init='uniform', bias=True, output_activation='softmax', regression=False):
    if regression: output_activation = 'relu'
    input_layer = [Dense(sizes[0], input_dim=ninputs, activation=act, kernel_initializer=init, use_bias=bias)]
    output_layer = [Dense(noutputs, activation=output_activation, kernel_initializer=init, use_bias=bias)]
    hidden_layers = [Dense(size, activation=act, kernel_initializer=init, use_bias=bias) for size in sizes[1:]]
    dnn = Sequential(input_layer + hidden_layers + output_layer)
    dnn.compile(loss=('categorical_crossentropy' if not regression else 'mape'), optimizer='adam', metrics=(['acc'] if not regression else []))
    dnn.summary()
    return dnn


def load_train_data(fn, num_ys = 4):
    data = pd.read_csv(fn, sep=';', header=0, index_col=0)
    ncols = len(data.columns)
    return data.iloc[:, :ncols - num_ys], data.iloc[:, ncols - num_ys:]


if __name__ == '__main__':
    np.random.seed(23)
    xs, ys = load_train_data('whoisbetter.csv', 3)

    model_fn = 'trained_model.h5'

    if os.path.isfile(model_fn):
        dnn = load_model(model_fn)
    else:
        dnn = build_network(ninputs=len(xs.columns), noutputs=len(ys.columns), sizes=[190,64,32,16], regression=False)
        dnn.fit(xs, ys, validation_split=0.1, batch_size=10, epochs=1000, verbose=2)
        dnn.save(model_fn)

    res = dnn.predict(xs)
    best_actual = np.argmax(ys.values, axis=1)
    best_pred = np.argmax(res, axis=1)
    diff = np.subtract(best_actual, best_pred)
    print(best_pred)
    print(best_actual)
    print(diff)
    print(f'accuracy: {1-np.sum(np.abs(diff)) / len(diff)}')