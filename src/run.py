from controller import Controller
from numpy import inf
import time
import robobo
import numpy as np
import pickle as pkl
import sys
import os
import json
import codecs
import signal
np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.2f}'.format})

def sigmoid_activation(x):
	return 1. / (1. + np.exp(-x))

n_hidden_neurons = 0 # preceptron
port = 19997
n_out = 3
step_size_ms = 500
full_speed = 50
penalize_backwards = True
sim_length_s = 60.0
hardware = False
brain = "brains/perceptron_elitism_fresh_genes_port19997_Tournament/gen_9/9422.json" #"brains/gen_8/11071.json"
print("Test")
brain_2 = "brains/gen_8/11071.json"
var = 0.5

lower_green = np.array([36,25,25])
upper_green = np.array([86,255,255])
img_hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(img_hsv, lower_green, upper_green)
imask = mask > 0
img_green = np.zeros_like(x, np.uint8)
img_green[imask] = x[imask]
cv2.imwrite("g.png", img_green)


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

			output = sigmoid_activation(inputs.dot(weights) + bias)[0]
			out = inputs.dot(weights) + bias
		print("OUT::\n"+str(output))
		print("OUT RAW::\n"+str(out))
		# takes decisions about robobos actions
		left = full_speed * output[0]
		right = full_speed * output[1]
		punish = 0
		if penalize_backwards:
			punish = 5
		if self.n_out == 3:
			if output[2] > 0.5:
				left  = -left + punish
				right = -right + punish
		return left, right



def terminate_program(signal_number, frame):
	print("Ctrl-C received, terminating program\n\n")
	sys.exit(1)

rob = robobo.HardwareRobobo(camera=True).connect(address="10.15.3.52")#, port=19997)
"""
# signal.signal(signal.SIGINT, terminate_program)
# start_simulation(rob)
time.sleep(2)
# rob.play_simulation()
time.sleep(2)

with open(brain,"r") as f:
	x_ = np.array(json.load(f))

with open(brain_2,"r") as f:
	x_ = np.array(json.load(f))
import time
elapsed_time = 0
sim_length_ms = sim_length_s * 1000
input_init = input = -np.log(rob.read_irs()).astype(float)
while True:
	input = -np.log(rob.read_irs()).astype(float)
	input[input == inf] = 0.0
	print(input)
	for i, (x, y) in enumerate(zip(input, input_init)):
		if round(x, 0) == round(y, 0):
			input[i] = 0.0
	#input_init = input
	#if not hardware:
#		input = np.array([-x for x in input if x != 0])
	print("Observed:\n"+str(input))
	nn = player_controller(n_hidden_neurons, n_out)
	left, right = nn.control(input, x_)
	print("\nMovement:\nleft="+str(left)+"\nright="+str(right))
	rob.move(left, right, step_size_ms)
	#time.sleep(0.5)
	elapsed_time += step_size_ms

rob.stop_world()
"""
