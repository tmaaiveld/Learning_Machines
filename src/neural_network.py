from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import model_from_json


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


def save_nn(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


# def neural_net(num_sensors, params, optimizer='rms', load=''):
#
#     model = Sequential()
#
#     # First layer.
#     model.add(Dense(
#         params[0], init='lecun_uniform', input_shape=(num_sensors,)
#     ))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.2))
#
#     # Second layer.
#     model.add(Dense(params[1], init='lecun_uniform'))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.2))
#
#     # Output layer.
#     model.add(Dense(4, init='lecun_uniform')) #actions 4: forward, back, left, right
#     model.add(Activation('linear'))
#
#     if optimizer == 'rms':
#         opt = RMSprop()
#
#     elif optimizer == 'adam':
#         opt = Adam()
#
#     model.compile(loss='mse', optimizer=opt)
#
#     if load:
#         model.load_weights(load)
#
#     return model
