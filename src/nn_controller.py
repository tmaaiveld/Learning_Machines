import numpy as np

class Controller(object):

    def control_(self, params, cont = None):

        action1 = np.random.choice([1, 0])
        action2 = np.random.choice([1, 0])
        action3 = np.random.choice([1, 0])
        action4 = np.random.choice([1, 0])
        action5 = np.random.choice([1, 0])
        action6 = np.random.choice([1, 0])

        return [action1, action2, action3, action4, action5, action6]


class player_controller(Controller):

	def control(self, params,cont):

		params = (params-min(params))/float((max(params)-min(params))) # standardizes
		params = np.hstack( ([1.], params) )
		n_outs = [5] # number of output neurons (sprite actions)
		n_params = [len(params)] # number of input variables
		n_hlayers = 0	 # number of hidden layers
		n_lneurons = [] # number of neurons in each hidden layer
		neurons_layers = []  # array of ann layers.
		neurons_layers.append(params) # adds input neurons layer to the ann.

		weights = cont # reads weights of solution

		if n_hlayers==1:
			nh = n_params[0]*n_lneurons[0]
			weights1 = weights[:nh].reshape( (n_params[0], n_lneurons[0]) )
			output1 = 1./(1. + np.exp(-params.dot(weights1)))
			output1 = np.hstack( ([1],output1) )
			weights2 = weights[nh:].reshape( (n_lneurons[0]+1, n_outs[0]) )
			output = 1./(1. + np.exp(-output1.dot(weights2)))
		else:

			weights = weights.reshape( (n_params[0], n_outs[0]) )
			output = 1./(1. + np.exp(-params.dot(weights)))


		# takes decisions about sprite actions

		if output[0] > 0.5:
			left = 1
		else:
			left = 0

		if output[1] > 0.5:
			right = 1
		else:
			right = 0

		if output[2] > 0.5:
			jump = 1
		else:
			jump = 0

		if output[3] > 0.5:
			shoot = 1
		else:
			shoot = 0

		if output[4] > 0.5:
			release = 1
		else:
			release = 0


		return [left, right, jump, shoot, release]

