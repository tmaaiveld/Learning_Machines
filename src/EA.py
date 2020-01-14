# imports framework
import sys
from deap import creator, base, tools, algorithms
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

actions = {'forward': (30.0, 30.0),
           'left': (10.0, 20.0),
           'right': (20.0, 10.0),
           'sharp left': (10.0, 30.0),
           'sharp right': (30.0, 10.0)
           }  # 'backward': (-25,-25)

#actions = {'forward': (40, 40),
#           'left': (15, 30),
#           'right': (30, 15),
#           'sharp left': (-20, 20),
#           'sharp right': (20, -20)
#           }  # 'backward': (-25,-25)

hardware = False
port = 19997

n_hidden_neurons = 10

step_size_ms = 250
sim_length_s = 60
kill_on_crash = True

dom_u = 1
dom_l = -1
npop = 10
gens = 10
mutation = 0.05
last_best = 0
cross_prob = 0.5
start_pop = 0
recovery_mode = False

if hardware:
	rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7")
else:
	rob = robobo.SimulationRobobo().connect(address='172.17.0.1', port=port) # 19997
	rob.stop_world()
	time.sleep(0.1)

def eval(x):
	global experiment_name
	global gen
	print("starting evaluation")

	signal.signal(signal.SIGINT, terminate_program)
	# start_simulation(rob)
	time.sleep(2)
	rob.play_simulation()
	time.sleep(2)

	elapsed_time = 0
	fitness = 0
	last_position = np.array([0,0,0])
	sim_length_ms = sim_length_s * 1000
	while sim_length_ms > elapsed_time:
		print("--------------------------\nElapsed time: "+str(elapsed_time))
		input = np.log(rob.read_irs()).astype(float)
		input[input == -inf] = 0
		print("Observed:\n"+str(input))

		nn = player_controller(n_hidden_neurons)

		left, right = actions[nn.control(input, np.array(x))]
		print("\nMovement:\nleft="+str(left)+"\nright="+str(right))
		rob.move(left, right, step_size_ms)
		elapsed_time += step_size_ms
		crashed, last_position = detect_crash(rob, input, last_position)
		#if not crashed:
		fitness += get_fitness(left, right, input)
		print("Total Fitness: "+str(fitness))
		#else:

			# Penalize crash according to remaining duration
		#	fitness = -((sim_length_ms - elapsed_time) * 10) + fitness
		#	print("penalizing with:"+str(fitness))
			# Penalize by ending the episode early			
			#break
			# Penalize by giving no fitness
			#print("Penalizing crash with no fitness")
			#pass
	print("Evaluation done, final fitness:"+str(fitness))
	print("--------------------------")
	rob.stop_world()
	# np.savetxt(experiment_name_new+str(int(fitness))+".txt",np.array(x))
	json_file = experiment_name_new+str(int(fitness))+".json"
	json.dump(x,codecs.open(json_file, "w", encoding="utf-8"), indent = 4)
	file_fit = open(experiment_name + 'results.txt', 'a')
	file_fit.write('\n' + str(gen) + ' ' + str(round(fitness, 6)))
	file_fit.close()
	return (fitness,)

def detect_crash(rob, input, last_position):
	current_position = np.array(rob.position())
	sensor_bound = -4
	position_bound = .01
	print("last position: "+str(last_position))
	print("current position: "+str(current_position))
	dist = np.linalg.norm(last_position - current_position)
	print("Detecting crash:\nsensor\n"+str(input)+"\n\ndist:\n"+str(dist))
	if min(input[3:] / 10) < sensor_bound or dist < position_bound:
		print("CRASH!")
		return True, current_position
	return False, current_position

def get_fitness(left, right, input):
	s_trans = left + right
	rot_max = 20  # from (0,20)
	rot_min = 0   # (30,30)
	# Normalized rotation
	s_rot = float((abs(left - right) - rot_min) / (rot_max - rot_min))
	v_max = 0
	v_min = -35
	v_sens = (sum(input[3:]) - v_min) / (v_max - v_min)
	print("")
	print("Fitness: ")
	print("s_trans "+str(s_trans))
	print("s_rot "+str(s_rot))
	print("v_sens "+str(v_sens))
	fit = s_trans * (1-s_rot) * (v_sens)
	print("total: "+str(fit))
	print("")	
	return fit

def terminate_program(signal_number, frame):
	print("Ctrl-C received, terminating program\n\n")
	sys.exit(1)


def initIndividual(icls, content):
    return icls(content)

def initPopulation(pcls, ind_init, filename):
    with open(filename, "r") as pop_file:
        contents = json.load(pop_file)
    return pcls(ind_init(c) for c in contents)

def sigmoid_activation(x):
	return 1. / (1. + np.exp(-x))


# implements controller structure for robobo
class player_controller(Controller):
	def __init__(self, _n_hidden):
		# Number of hidden neurons
		self.n_hidden = [_n_hidden]

	def control(self, inputs, controller):
		# Normalises the input using min-max scaling
		inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))

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
		else:
			bias = controller[:5].reshape(1, 5)
			weights = controller[5:].reshape((len(inputs), 5))

			output = sigmoid_activation(inputs.dot(weights) + bias)[0]

		# takes decisions about robobos actions
		ind = np.argmax(output)
		action = actions.keys()[ind]
		return action


selections = {#"NSGA2": tools.selNSGA2,
              "Tournament": tools.selTournament}

for selection in selections.keys():

	this_selection = selections[selection]
	# number of weights for multilayer with 10 hidden neurons
	#
	num_sensors = 8
	n_vars = (num_sensors+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

	# n_vars = (num_sensors() + 1) * 5  # perceptron
	# n_vars = (num_sensors()+1)*10 + 11*5  # multilayer with 10 neurons
	# n_hidden = 50
	# n_vars = (num_sensors()+1)*n_hidden + (n_hidden+1)*5 # multilayer with 50 neurons

	experiment_name = "experiments_port"+str(port)+"_" + selection+"/"
	if not os.path.exists(experiment_name):
		os.makedirs(experiment_name)
	else:
		recovery_mode = True

	# n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5 # multilayer with 50 neurons

	# DEAP Implementation
	stats = tools.Statistics(key=lambda ind: ind.fitness.values)
	stats.register("avg", np.mean, axis=0)
	stats.register("std", np.std, axis=0)
	stats.register("min", np.min, axis=0)
	stats.register("max", np.max, axis=0)
	logbook = tools.Logbook()
	creator.create("FitnessMax", base.Fitness, weights=(1.0,))
	creator.create("Individual", list, fitness=creator.FitnessMax)

	toolbox = base.Toolbox()

	toolbox.register("attr_float", np.random.uniform, dom_l, dom_u)
	toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_vars)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)

	toolbox.register("evaluate", eval)
	toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=dom_l, up=dom_u, eta=30.0)
	toolbox.register("mutate", tools.mutPolynomialBounded, low=dom_l, up=dom_u, eta=20.0, indpb=1.0 / n_vars)
	toolbox.register("select", this_selection)

	population = toolbox.population(n=npop)
	i = 0

	for gen in range(gens):
		experiment_name_new = experiment_name + "gen_" + str(gen) + "/"
		if not os.path.exists(experiment_name_new):
			os.makedirs(experiment_name_new)
		elif len(os.listdir(experiment_name_new)) > 0:
			recovery = True
		if recovery_mode:
			print("------------\nRECOVERY")
			ls = os.listdir(experiment_name_new)
			all_data = []
			for path in ls:
				if "all_data.json" in path:
					continue
				with open(experiment_name_new+path) as f:
					all_data.append(json.load(f))
			json_file = "all_data.json"
			json.dump(all_data,codecs.open(experiment_name_new+json_file, "w", encoding="utf-8"), indent = 4)
			# Recover old genotypes
			toolbox.register("individual_guess", initIndividual, creator.Individual)
			toolbox.register("population_guess", initPopulation, list, toolbox.individual_guess, experiment_name_new+"all_data.json")


			population_old = toolbox.population_guess()
			fits_old = [(float(x.replace(".json","")),) for x in ls if "all_data" not in x]	
	
			print("LENGTH pop old:"+str(len(population_old)))
			if len(population_old) >= gens:
				print("Skipping generation "+str(gen))
				population = population_old
				recovery_mode = False
				continue
			population = population[len(population_old):]
			# Recover old fitness values
							
		offspring = algorithms.varAnd(population, toolbox, cxpb=cross_prob, mutpb=mutation)
		#offspring = offspring + population
		fits = toolbox.map(toolbox.evaluate, offspring)

		if recovery_mode:
			fits = fits + fits_old
			population = offspring + population_old

		print("Fits", fits)
		for fit, ind in zip(fits, offspring):
			ind.fitness.values = fit
		if selection == "NSGA2":
			population = toolbox.select(offspring+population, k=npop)
		elif selection == "Tournament":
			population = toolbox.select(offspring+population, k=npop, tournsize=3)
		best = tools.selBest(population, k=1)

		record = stats.compile(population)

		# saves simulation state
		logbook.record(gen=gen, evals=30, **record)
		solutions = [offspring, fits]

		best_fit = logbook.select("max")[0][0]

		# saves results
		file_aux = open(experiment_name + 'results.txt', 'a')
		print('\n GENERATION ' + str(gen) + ' ' + str(round(best_fit, 6)))
		file_aux.write('\n' + str(gen) + ' ' + str(round(best_fit, 6)))
		file_aux.close()
		
	#with open(experiment_name + "logbook.pkl", "wb") as f:
#		pkl.dump(logbook, f)
#	with open(experiment_name + "best.pkl", "wb") as f:
#		pkl.dump(best, f)
#	np.savetxt("best.txt", best)
print("Done!")
