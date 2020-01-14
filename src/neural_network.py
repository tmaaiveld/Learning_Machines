from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation
from keras.models import model_from_json
import numpy as np


def init_nn(input_dims, output_dims, hidden_layers, from_file):

    if not from_file:
        model = Sequential()

        model.add(Dense(hidden_layers[0], input_dim=input_dims, activation='linear'))
        model.add(Dropout(0.2))

        model.add(Dense(hidden_layers[1], activation='linear'))
        model.add(Dropout(0.2))

        model.add(Dense(output_dims, activation='linear'))
        print("Initialized new model, compiling...")

    else:
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("model.h5")
        print("Loaded model from disk, compiling...")

    model.compile(loss='MSE', optimizer='adam')

    return model


def init_nn_EC(input_dims, output_dims, ind, init_bias=False):
    """Initialize the neural network for the ES implementation."""

    if not init_bias:
        weights = [np.array(ind[:-2], dtype='float32').reshape(8, 2),
                   np.array(ind[-2:], dtype='float32')]
    else:
        weights = [np.array(ind[:-2], dtype='float32').reshape(8, 2),
                   np.array([1., 1.], dtype='float32')]

    model = Sequential()
    model.add(Dense(output_dims, input_dim=input_dims, weights=weights, activation='tanh'))
    model.add(Dropout(0.2))
    return model


def save_nn(model, path):
    # serialize model to JSON
    model_json = model.to_json()
    with open(path + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filepath=path + ".h5")
    print("Saved final NN to disk")
