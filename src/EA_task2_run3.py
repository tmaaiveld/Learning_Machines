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

np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.2f}'.format})

from camera import Camera

# import imutils

hardware = False
run = 3
port = 19998 - run
kill_on_crash = False
base_name = "experiments/test_food_foraging_run{}".format(run)
full_speed = 40
if kill_on_crash:
    base_name += "_killoncrash"
base_name += "_port" + str(port)
penalize_backwards = False
activation = 'tanh'

n_hidden_neurons = 0
num_sensors = 3 + 3 #+ 3 + 3 + 3
n_out = 2
step_size_ms = 400
sim_length_s = 90.0
max_food = 7.0
collected_food = 0.0
sensitivity = 30
n_vars = (num_sensors + 1) * n_out  # Simple perceptron

dom_u = 1
dom_l = -1
npop = 10
gens = 20
mutation = 0.1
cross_prob = 0.5
recovery_mode = False

if hardware:
    rob = robobo.HardwareRobobo(camera=True).connect(address="10.15.3.42")
else:
    # rob = robobo.SimulationRobobo().connect(address='172.20.10.3', port=port)  # 19997
    rob = robobo.SimulationRobobo().connect(address='192.168.1.70', port=port)  # 19997

def eval(x):
    global experiment_name
    global gen
    print("starting evaluation")
    if not hardware:
        rob.stop_world()
    #	time.sleep(0.1)
    signal.signal(signal.SIGINT, terminate_program)
    # start_simulation(rob)
    #	time.sleep(2)
    if not hardware:
        rob.play_simulation()
        rob.set_phone_tilt(119.75, 1.0)  # tilting didn't seem to make sense in the simulation
    else:
        rob.set_phone_tilt(90, 4.0)
        time.sleep(5)

    #	time.sleep(2)

    elapsed_time = 0
    fitness = 0
    positions = []
    last_position = np.array([0, 0, 0])
    sim_length_ms = sim_length_s * 1000
    food_old = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])
    collected_food = rob.collected_food()

    nn = player_controller(n_hidden_neurons, n_out)

    step = 0
    while sim_length_ms > elapsed_time and collected_food is not 7:
        start_time = time.time()
        print("\n--------------------------\nElapsed time: " + str(elapsed_time) + "\n")
        # input = np.log(rob.read_irs()).astype(float)
        # input[input == -inf] = 0.0

        image = rob.get_image_front()
        img = Camera(image)
        food = np.array(img.capture_food_image(sensitivity, step))

        print('food:')
        print(food)

        new_input = np.vstack([food, food_old[:num_sensors/3 - 1]])
        print(new_input)

        collected_food = rob.collected_food()
        print('food collected', collected_food)

        left, right = nn.control(new_input, np.array(x))
        print("\nMovement:\nleft={}\nright={}".format(left, right))
        rob.move(left, right, step_size_ms)
        # time.sleep(250.0/1000.0)
        food_old = new_input
        elapsed_time += step_size_ms
        # crashed, last_position = detect_crash(rob, input, last_position)
        positions.append(last_position)

        obj_seen = not np.array_equal(food, np.array((0, 0, 1)))
        print('Object seen: {}'.format(obj_seen))

        fitness += get_fitness_foraging(left, right, obj_seen)  # , input)
        print("Total Fitness: {0:.2f}".format(fitness))

        step += 1

    print("Evaluation done, final fitness: {}".format(fitness))
    print("--------------------------")

    # Weigh fitness by collected food
    food_factor = collected_food / max_food
    fitness_final = food_factor * fitness / step * 100
    print("final fitness: {}".format(fitness_final))

    if not hardware:
        rob.stop_world()

    print("Evaluation of the individual done")
    print("fitness: {}".format(fitness))
    print("final food collected: {} of 7".format(collected_food))
    print("food penalty factor: {}".format(food_factor))
    print("scaled fitness: {}".format(fitness_final))

    # np.savetxt(experiment_name_new+str(int(fitness))+".txt",np.array(x))
    json_file = experiment_name_new + str(int(fitness_final)) + ".json"
    i = 0

    while os.path.exists(json_file):
        i += 1
        json_file = experiment_name_new + str(int(fitness_final + i)) + ".json"
        print("Renaming duplicate: {}".format(fitness_final))

    json.dump(x, codecs.open(json_file, "w", encoding="utf-8"), indent=4)
    file_fit = open(experiment_name + 'results.txt', 'a')
    file_fit.write('\n' + str(gen) + ' ' + str(round(fitness_final, 6))
                   + ' ' + str(collected_food) + ' ' + str(step))
    file_fit.close()
    file_pos = open(experiment_name + 'positions.txt', 'a')
    file_pos.write('\n' + str(gen) + ' ' + str(positions))
    file_pos.close()
    return (fitness_final,)


def detect_crash(rob, input, last_position):
    current_position = np.array(rob.position())
    sensor_bound = -5.5
    position_bound = .001
    print("last position: {}".format(last_position))
    print("current position: {}".format(current_position))
    dist = np.linalg.norm(last_position - current_position)
    print("Detecting crash:\nsensor\n" + str(input) + "\n\ndist:\n" + str(dist))
    if min(input) < sensor_bound:  # or dist < position_bound:
        print("CRASH!")
        return True, current_position
    return False, current_position


def get_fitness(left, right, input):
    s_trans = left + right
    rot_max = 40  # from (0,20)
    rot_min = 0  # (30,30)
    # Normalized rotation
    s_rot = float((abs(left - right) - rot_min) / (rot_max - rot_min))
    v_max = 0
    v_min = -74
    v_sens = (sum(input) - v_min) / (v_max - v_min)
    print("")
    print("Fitness: ")
    print("s_trans {}".format(s_trans))
    print("s_rot {}".format(s_rot))
    print("v_sens {}".format(v_sens))
    fit = s_trans * (1 - s_rot) * (v_sens)
    print("total: {}".format(fit))
    print("")
    return fit


def get_fitness_foraging(left, right, obj_seen):
    s_trans = (left + right) / (2 * full_speed)
    rot_max = 2 * full_speed  # from (0,20)
    rot_min = 0  # (30,30)
    # Normalized rotation
    s_rot = float((abs(left - right) - rot_min) / (rot_max - rot_min))
    #	v_max = 0
    #	v_min = -74
    #	v_sens = (sum(input) - v_min) / (v_max - v_min)
    print("\nFitness: ")
    print("s_trans {}".format(s_trans))
    print("s_rot {}".format(s_rot))
    #	print("v_sens "+str(v_sens))

    if obj_seen:
        fit = s_trans * (1 - s_rot)
    else:
        fit = 0

    print("total: " + str(fit))
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


def tanh_activation(x):
    return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)


# implements controller structure for robobo
class player_controller(Controller):
    def __init__(self, _n_hidden, _n_out):
        # Number of hidden neurons
        self.n_hidden = [_n_hidden]
        self.n_out = _n_out

    def control(self, inputs, controller):

        inputs = inputs.flatten()

        if self.n_hidden[0] > 0:
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


# selections = {"NSGA2": tools.selNSGA2},
selections = {"Tournament": tools.selTournament}

for selection in selections.keys():

    this_selection = selections[selection]

    experiment_name = base_name + "_" + selection + "/"
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
    all_gens = population

    for gen in range(gens):
        experiment_name_new = experiment_name + "gen_" + str(gen) + "/"
        if not os.path.exists(experiment_name_new):
            os.makedirs(experiment_name_new)
        elif len(os.listdir(experiment_name_new)) > 0:
            print("listdir " + str(experiment_name_new) + " :", os.listdir(experiment_name_new))
            recovery_mode = True
        if recovery_mode:
            print("\n------------\nRECOVERY\n")
            ls = os.listdir(experiment_name_new)
            all_data = []
            for path in ls:
                if "all_data.json" in path:
                    continue
                with open(experiment_name_new + path) as f:
                    all_data.append(json.load(f))
            json_file = "all_data.json"
            json.dump(all_data, codecs.open(experiment_name_new + json_file, "w", encoding="utf-8"), indent=4)
            # Recover old genotypes
            toolbox.register("individual_guess", initIndividual, creator.Individual)
            toolbox.register("population_guess", initPopulation, list, toolbox.individual_guess,
                             experiment_name_new + "all_data.json")

            population_old = toolbox.population_guess()
            print("Loaded {} individuals from current generation".format(len(population_old)))
            fits_old = [(float(x.replace(".json", "")),) for x in ls if "all_data" not in x]

            if len(population_old) >= npop:
                print("Skipping generation {}".format(gen))
                population = population_old
                recovery_mode = False
                continue

            population = population[len(population_old):]

        population = algorithms.varAnd(population, toolbox, cxpb=cross_prob, mutpb=mutation)

        # offspring = offspring + population
        fits = toolbox.map(toolbox.evaluate, population)

        if recovery_mode:
            fits = fits + fits_old
            population = population + population_old

        print("Fits", fits)
        for fit, ind in zip(fits, population):
            ind.fitness.values = fit
        if selection == "NSGA2":
            population = toolbox.select(population, k=npop - 1)
        elif selection == "Tournament":
            population = toolbox.select(population, k=npop - 1, tournsize=3)

        all_gens += population
        elite = tools.selBest(all_gens, k=1)

        population += elite

        record = stats.compile(population)

        # population = best_50 + toolbox.population(n=npop/2)

        # saves simulation state
        logbook.record(gen=gen, evals=30, **record)
        solutions = [population, fits]

        best_fit = logbook.select("max")[0][0]

        # saves results
        file_aux = open(experiment_name + 'results.txt', 'a')
        print('\n GENERATION ' + str(gen)) # + ' ' + str(round(best_fit, 6)))
        # file_aux.write('\n' + str(gen) + ' ' + str(round(best_fit, 6)))
        file_aux.close()

# with open(experiment_name + "logbook.pkl", "wb") as f:
#		pkl.dump(logbook, f)
#	with open(experiment_name + "best.pkl", "wb") as f:
#		pkl.dump(best, f)
#	np.savetxt("best.txt", best)
print("Done!")
