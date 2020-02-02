# imports framework
import json
# imports other libs
import time

import numpy as np
import robobo

from controller import Controller

np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.2f}'.format})

from camera import Camera

# import imutils

hardware = True
port = 19997
kill_on_crash = False
base_name = "experiments/test_food_foraging"
full_speed = 80
if kill_on_crash:
    base_name += "_killoncrash"
base_name += "_port" + str(port)
penalize_backwards = False
activation = 'tanh'
brain="src/115.json"

n_hidden_neurons = 0
num_sensors = 3 + 3
n_out = 2
step_size_ms = 400
sim_length_s = 200.0
max_food = 7.0
collected_food = 0.0
sensitivity = 30
n_vars = (num_sensors + 1) * n_out  # Simple perceptron
# n_vars = (num_sensors()+1)*10 + 11*5  # multilayer with 10 neurons

dom_u = 1
dom_l = -1
npop = 10
gens = 21
mutation = 0.
cross_prob = 0.
recovery_mode = False


rob = robobo.HardwareRobobo(camera=True).connect(address="10.15.3.52")

rob.set_phone_tilt(90, 4.0)
time.sleep(5)

with open(brain, "r") as f:
    x=json.load(f)

def sigmoid_activation(x):
    return 1. / (1. + np.exp(-x))


def tanh_activation(x):
    return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)


# implements controller structure for robobo
class player_controller(Controller):
    def __init__(self, _n_hidden, _n_out):
        # Number of hidden neurons
        self.n_hidden = [_n_hidden]
        self.n_out = _n_out

    def control(self, inputs, controller):
        # Normalises the input using min-max scaling
        #		inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))
        #		print("Inputs NN:"+str(inputs))

        if self.n_hidden[0] > 0:
            # Preparing the weights and biases from the controller of layer 1

            # Biases for the n hidden neurons
            bias1 = controller[:self.n_hidden[0]].reshape(1, self.n_hidden[0])
            # Weights for the connections from the inputs to the hidden nodes
            weights1_slice = len(inputs) * self.n_hidden[0] + self.n_hidden[0]
            weights1 = controller[self.n_hidden[0]:weights1_slice].reshape((len(inputs), self.n_hidden[0]))

            # Outputs activation first layer.
            output1 = sigmoid_activation(inputs.dot(weights1) + bias1)

            # Preparing the weights and biases from the controller of layer 2
            bias2 = controller[weights1_slice:weights1_slice + self.n_out].reshape(1, self.n_out)
            weights2 = controller[weights1_slice + self.n_out:].reshape((self.n_hidden[0], self.n_out))

            # Outputting activated second layer. Each entry in the output is an action
            output = sigmoid_activation(output1.dot(weights2) + bias2)[0]
            out = output1.dot(weights2) + bias2
        else:
            bias = controller[:self.n_out].reshape(1, self.n_out)
            weights = controller[self.n_out:].reshape((len(inputs), self.n_out))

            print('\n currently testing: \n{}\n{}'.format(weights, bias))

            if activation == 'tanh':
                output = tanh_activation(inputs.dot(weights) + bias)[0]
            elif activation == 'sigmoid':
                output = sigmoid_activation(inputs.dot(weights) + bias)[0]

            out = inputs.dot(weights) + bias

        print("OUT::\n" + str(output))
        print("OUT RAW::\n" + str(out))
        # takes decisions about robobos actions
        left = full_speed * output[0]
        right = full_speed * output[1]

        punish = 5 if penalize_backwards else 0

        if self.n_out == 3:
            if output[2] > 0.5:
                left = -left + punish
                right = -right + punish

        return left, right


nn = player_controller(n_hidden_neurons, n_out)
food_old = np.array((0, 0, 128))
step = 0

while True:
    image = rob.get_image_front()
    img = Camera(image)
    food = np.array(img.capture_food_image(sensitivity, step))

    print('food:')
    print(food)

    # new_input = 2.5 * (np.array(list(food_old) + list(food)))

    new_input = (np.array(list(food_old) + list(food)))
    new_input[0], new_input[3] = 2. * new_input[0], 2. * new_input[3]  # scale left screen distance
    new_input[1], new_input[4] = 4. * new_input[1], 4. * new_input[4]  # scale right screen distance
    # time.sleep(1)
    print('inputs: \n{}'.format(new_input.reshape((2, 3))))

    # collected_food = rob.collected_food()
    # print('food collected', collected_food)

    left, right = nn.control(new_input, np.array(x))
    print("\nMovement:\nleft={}\nright={}".format(left, right))
    rob.move(left, right, step_size_ms)
    # time.sleep(250.0/1000.0)
    food_old = food

    obj_seen = not np.array_equal(food, np.array((0, 0, 1)))
    step = + 1
