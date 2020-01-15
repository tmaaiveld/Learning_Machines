actions = {'forward': (30.0, 30.0),
           'left': (10.0, 20.0),
           'right': (20.0, 10.0),
           'sharp left': (10.0, 30.0),
           'sharp right': (30.0, 10.0)
           }  # 'backward': (-25,-25)

from controller import Controller

# imports other libs
from numpy import inf
import time
import robobo
import numpy as np
import pickle as pkl
import os
import json 
import codecs
import signal
np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.2f}'.format})

def sigmoid_activation(x):
	return 1. / (1. + np.exp(-x))

n_hidden_neurons = 10
port = 19997
step_size_ms = 250
sim_length_s = 30.0
hardware = False

class player_controller(Controller):
	def __init__(self, _n_hidden):
		# Number of hidden neurons
		self.n_hidden = [_n_hidden]

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
			bias2 = controller[weights1_slice:weights1_slice + 5].reshape(1, 5)
			weights2 = controller[weights1_slice + 5:].reshape((self.n_hidden[0], 5))

			# Outputting activated second layer. Each entry in the output is an action
			output = sigmoid_activation(output1.dot(weights2) + bias2)[0]
			out = output1.dot(weights2) + bias2
		else:
			bias = controller[:5].reshape(1, 5)
			weights = controller[5:].reshape((len(inputs), 5))

			output = sigmoid_activation(inputs.dot(weights) + bias)[0]
			out = inputs.dot(weights) + bias
		print("OUT::\n"+str(output))
		print("OUT RAW::\n"+str(out))
		# takes decisions about robobos actions
		ind = np.argmax(output)
		print("index:"+str(ind))
		action = actions.keys()[ind]
		print("action:"+str(action))
		return action


def terminate_program(signal_number, frame):
	print("Ctrl-C received, terminating program\n\n")
	sys.exit(1)

if hardware:
	rob = robobo.SimulationRobobo().connect(address='172.17.0.1')#, port=port) # 19997
else:
	rob = robobo.HardwareRobobo(camera=False).connect(address="192.168.43.176")

signal.signal(signal.SIGINT, terminate_program)
# start_simulation(rob)
time.sleep(2)
# rob.play_simulation()
time.sleep(2)

with open("experiments_#0_elitism_port19998_NSGA2/gen_9/2033.json","r") as f:
	x = json.load(f) 
elapsed_time = 0
sim_length_ms = sim_length_s * 1000
input_init = input = -np.log(rob.read_irs()).astype(float)
while sim_length_ms > elapsed_time:
	input = -np.log(rob.read_irs()).astype(float)
	input[input == -inf] = 0.0
	for i, (x, y) in enumerate(zip(input, input_init)):
		if x == y:
			input[i] = 0.0
	#if not hardware:
#		input = np.array([-x for x in input if x != 0])
	print("Observed:\n"+str(input))
	nn = player_controller(n_hidden_neurons)
	left, right = actions[nn.control(input, np.array(x))]
	print("\nMovement:\nleft="+str(left)+"\nright="+str(right))
	rob.move(left, right, step_size_ms)
	elapsed_time += step_size_ms

rob.stop_world()
