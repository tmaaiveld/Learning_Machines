"""
Current approach:
(1+1) ES Algorithm. The algorithm generates individuals (lists of neural network weights coupled with a mutation
vector) and runs them through a simulation to evaluate them. After evaluation, the fittest individual is selected
and mutated to produce a new offspring, which is evaluated in turn, etcetera. The mutation parameters for the NN
weights are evolved over time as well.

todo's
- toy with initialization parameters, different mazes
- Visualization
    > show increase of reward over time
    > increase in wheel speeds over time?
    > any other metrics showing improvement over iterations
- Implementing camera features
- Testing on hardware
- Improving neural network? Should the neural network contain hidden layers to model non-linear properties?
- Thoroughly check ES algorithm -> Is the 1/5 rule being followed? Does the mutation function properly?
    > adapting mutation parameters?
    > rereading Eiben book or Schwefer 2002 might be helpful
- Could evolve sensor arrangement as well
- Streamlining driving (smoother drive)
- Implement crossover to escape getting stuck? (might shake things up a bit)
    > perhaps a structure where X-over happens when fitnesses are very close.
- performing evolutionary computing on what sensors are active? See Eiben 2015. Might help solve maze problem, as the
  sensor arrangement is pretty crucial
- could gather all params in a dict to prettify code
"""

from __future__ import print_function

import time
from datetime import datetime
import random
import array
import numpy as np
import sys
import robobo
import signal
from os import listdir

# import cv2
# import prey

from neural_network import init_nn_EC

try:
    from deap import base
    from deap import creator
    from deap import tools
    import pandas as pd
except ModuleNotFoundError:
    raise ModuleNotFoundError('Run `pip install deap` and `pip install pandas` first.')

np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.2f}'.format})


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program\n\n")
    sys.exit(1)


def save_model_at(model_path, fitness, episode, sim_number):

    prev_model_number = [f.split('_')[1] for f in listdir(model_path)][-1] if listdir(model_path) else -1

    if episode == 0:
        model_number = int(prev_model_number) + 1
    else:
        model_number = prev_model_number

    model_name = "model_" + str(model_number) +  "_sim" + str(sim_number)

    return model_path + model_name


def save_data_at(data_path, data, episode):

    data_columns = ['weight_1_' + str(i + 1) for i in range(8)] + \
                   ['weight_2_' + str(i + 1) for i in range(8)] + \
                   ['bias_' + str(i + 1) for i in range(2)] +     \
                   ['avg_s_trans', 'avg_s_rot', 'avg_v_sens'] +   \
                   ['episode_time_ms', 'time_of_day', 'distance_to_previous_model', 'fitness']

    data = pd.DataFrame(data, columns=data_columns)
    date = str(datetime.today().day)

    sessions_today = [int(f.split("_")[1]) for f in listdir(data_path) if f.split("_")[2] == date]

    if episode == 0:
        model_index = max(sessions_today) + 1 if listdir(data_path) else 0
    else:
        model_index = max(sessions_today) if listdir(data_path) else 0

    data.to_csv(data_path + "data_" + str(model_index) + "_" + date + "_jan.csv")


def generateES(icls, scls, size, imin, imax, smin, smax, MODEL_PATH=False):
    """Generate neural network weights ('individuals') and evolution parameters ('strategies')"""
    if not MODEL_PATH:

        ind = icls([random.uniform(imin, imax) for _ in range(size - 2)] +
                   [1. for _ in range(2)])

        print(ind)
        print("Initializing random model...")

    else:
        prev_model = pd.read_csv(MODEL_PATH, header=None, squeeze=True)
        print("Reinitializing model from " + MODEL_PATH + "...")
        ind = icls(prev_model)

    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind


def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children

        return wrappper

    return decorator


def evaluate(data, recovery_time):
    """Fitness as implemented by Eiben et al. 2015"""
    return (data['s_trans'] * (1 - data['s_rot']) * (1 - data['v_sens']))[recovery_time:].sum()


def main():
    SENS_NAMES = ["IR" + str(i + 1) for i in range(8)]

    EP_COUNT = 500
    STEP_COUNT = list(range(10,100,2)) + [100] * EP_COUNT
    STEP_SIZE_MS = 500

    # ES parameters
    M_MAX = 20  # sets the maximum speed of the robot
    MUT_PROB_0 = 0.95  # sets the probability of mutating an allele (p_m) at t=0, which decreases exponentially
    C = 2.0  # sets the rate at which mutation step size cools down... I think? See Schwefer 2002.
    MIN_VALUE, MAX_VALUE = -1., 1.  # random init range for NN weights (positive to ensure driving forward)
    MIN_STRATEGY, MAX_STRATEGY = -1., 1.
    RECOVERY_TIME = 5  # sets the number of initial steps not evaluated

    # Initialization -> SET THESE PARAMETERS CAREFULLY SO YOU DON'T OVERWRITE YOUR WORK.
    hardware = False
    learning = True
    load_model = False
    save_model = True
    save_data = True

    MODEL_PATH = "src/models/"
    DATA_PATH = "src/data/"
    SIM_NUMBER = 0  # [0,1,2] -> box, pillars, maze
    CURRENT_TIME = "".join([char if char.isalnum() else "_" for char in str(datetime.today())[5:][:-10]])

    if hardware:
        rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7")
    else:
        rob = robobo.SimulationRobobo(number=["","#2","#0"][SIM_NUMBER]).connect(address='172.20.10.3', port=19997)


    # DEAP initialization
        # genotype structure: first 16 are weights (2 for each input, last allele is bias node gene)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode="d",
                   fitness=creator.FitnessMax, strategy=None)
    creator.create("Strategy", array.array, typecode="d")

    toolbox = base.Toolbox()
    toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
                     (2 * len(SENS_NAMES) + 2), MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)

    toolbox.register("mutate", tools.mutESLogNormal, c=C)
    toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))

    ind = toolbox.individual(MODEL_PATH=MODEL_PATH) if load_model else toolbox.individual()
    pop = [ind, ind]
    fitnesses = [-10000]

    data = []

    ########## MAIN ALGORITHM ##########

    for ep in range(EP_COUNT):
        ep_start_time = time.time()

        print('\n--- episode {} ---'.format(ep + 1))

        signal.signal(signal.SIGINT, terminate_program)
        rob.play_simulation()

        model = init_nn_EC(input_dims=len(SENS_NAMES), output_dims=2,
                           ind=ind)

        print('Testing model parameters: \n', np.array(model.get_weights()))
        time.sleep(3)

        ########## EVALUATION ##########

        ep_data = []

        for i in range(STEP_COUNT[ep]):
            start_time = time.time()

            print('\n--- step {} ---\n'.format(i + 1))

            IR = -abs(np.log(np.array(rob.read_irs())))
            IR[np.isinf(IR)] = 0
            current_position = np.array(rob.position())
            wheels = model.predict(np.expand_dims(IR, axis=0))[0] * M_MAX

            print("ROB IRs: {}".format(IR / 10))
            print("robobo is at {}".format(current_position))
            print("Wheel speeds: ", wheels)

            # move the robot
            rob.move(wheels[0], wheels[1], STEP_SIZE_MS)

            # collect data
            s_trans = wheels[0] + wheels[1]
            s_rot = abs(wheels[0] - wheels[1]) / (2 * M_MAX)  # make readings negative for correct initialization
            v_sens = min(max(-IR) / 6.3, 1)  # ugly normalization

            ep_data.append([s_trans, s_rot, v_sens])

            # print_progress(i, STEP_COUNT[ep])
            step_time = time.time() - start_time

        ########## EVOLUTION ##########

        model_dist = 0  # set model distance to 0 if not learning
        if learning:

            ep_data = pd.DataFrame(ep_data, columns=('s_trans', 's_rot', 'v_sens'))

            # Calculate the fitness of the evaluated model
            ind_fit = round(evaluate(ep_data, RECOVERY_TIME) / STEP_COUNT[ep],3)
            ind.fitness.values = (ind_fit,)
            model_dist = round(((np.array(pop[0]) - np.array(ind))**2).sum(),3)

            print("this model's fitness (previous): ", ind_fit, ' (', fitnesses[0], ')')
            print("Euclidian distance to previous model: ", model_dist)

            # delete the inferior model
            fitnesses.append(ind_fit)
            print(fitnesses)

            del pop[1 - np.argmax(fitnesses)]
            del fitnesses[1 - np.argmax(fitnesses)]

            if save_model:
                best = pd.Series(list(pop[0]))
                best.to_csv(save_model_at(MODEL_PATH, fitnesses[0], ep, SIM_NUMBER), index=False)

            # Produce a new offspring
            mut_rate = 0.9 * (MUT_PROB_0**ep) + 0.1
            offspring = toolbox.clone(pop[0])
            ind, = toolbox.mutate(offspring, indpb=mut_rate)
            pop.append(ind)

        # save the episode data
        ep_time = time.time() - ep_start_time

        data.append(list(ind) +
                    list(np.array(ep_data).mean(axis=0)) +
                    [datetime.today()] + [ep_time] + [model_dist] + fitnesses)

        # build the episode data structure for statistics and write to .csv
        if save_data:
            save_data_at(DATA_PATH, data, ep)



if __name__ == "__main__":
    main()
