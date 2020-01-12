from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop, Adam


def init_nn(input_dims, output_dims, nn_dims):

    model = Sequential()

    model.add(Dense(nn_dims[0], input_dim=input_dims, activation='linear'))
    model.add(Dropout(0.2))

    model.add(Dense(nn_dims[1], activation='linear'))
    model.add(Dropout(0.2))

    model.add(Dense(output_dims, activation='linear'))

    model.compile(loss='MSE', optimizer='adam')

    return model




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