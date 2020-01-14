"""
Current approach:
Evolved NN to control wheel speeds (1+1 strategy)
...

todo's
- might need to implement a recovery time (see Eiben et al) while network is being switched (or stop the car).
- hyperNEAT to evolve network topology?
- Implement crossover to escape getting stuck? (might shake things up a bit)
- performing evolutionary computing on what sensors are active. See paper. Might help solve maze problem, as the
  sensor arrangement is pretty crucial



"""
from __future__ import print_function

import time
import random
import array
import numpy as np
import sys

try:
    from deap import base
    from deap import creator
    from deap import tools
    import pandas as pd
except ModuleNotFoundError:
    raise ModuleNotFoundError('Run `pip install deap` and `pip install pandas` first.')

from neural_network import init_nn_EC, save_nn

import robobo
import signal

# import cv2
# import prey

np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.2f}'.format})


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program\n\n")
    sys.exit(1)


def generateES(icls, scls, size, imin, imax, smin, smax, MODEL_PATH=False):
    """Generate neural network weights ('individuals') and evolution parameters ('strategies')"""
    if not MODEL_PATH:
        ind = icls(random.uniform(imin + 1, imax) for _ in range(size))
    else:
        ind = icls(pd.read_csv(MODEL_PATH))

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

    EP_COUNT = 1
    STEP_COUNT = [20, 40, 60, 80, 100, 150] + [200] * EP_COUNT
    STEP_SIZE_MS = 400

    M_MAX = 20
    MUT_PROB = 0.6
    C = 1.0
    MIN_VALUE, MAX_VALUE = -1., 1.
    MIN_STRATEGY, MAX_STRATEGY = -1., 1.
    RECOVERY_TIME = 5

    MODEL_PATH = "src/model.csv"  # set to False if not using a pre-trained model

    # Initialize the robot
    hardware = False
    learning = True
    load_model = False
    save_model = True

    if hardware:
        rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7")
    else:
        rob = robobo.SimulationRobobo().connect(address='172.20.10.3', port=19997)


    ### DEAP INITIALIZATION ###
    # genotype structure: first 16 are weights (2 for each input, last allele is bias node gene

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode="d",
                   fitness=creator.FitnessMax, strategy=None)
    creator.create("Strategy", array.array, typecode="d")

    toolbox = base.Toolbox()
    toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
                     (2 * len(SENS_NAMES) + 2), MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)

    toolbox.register("mutate", tools.mutESLogNormal, c=C, indpb=MUT_PROB)
    toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))

    ind = toolbox.individual(MODEL_PATH=MODEL_PATH) if load_model else toolbox.individual()
    pop = [ind, ind]
    fitnesses = [-10000]

    for episode in range(EP_COUNT):
        ep_start_time = time.time()

        print('\n--- episode {} ---'.format(episode + 1))

        ep_data = []

        signal.signal(signal.SIGINT, terminate_program)
        rob.play_simulation()

        weights = [np.array(ind[:-2], dtype='float32').reshape(8, 2),
                   np.array(ind[-2:], dtype='float32')]
        model = init_nn_EC(input_dims=len(SENS_NAMES), output_dims=2,
                           weights=weights)

        ########## EVALUATION LOOP ##########

        for i in range(STEP_COUNT[episode]):
            start_time = time.time()

            print('\n--- step {} ---\n'.format(i + 1))

            IR = np.log(np.array(rob.read_irs()))
            IR[np.isinf(IR)] = 0
            current_position = np.array(rob.position())
            wheels = model.predict(np.expand_dims(IR, axis=0))[0] * M_MAX

            print("ROB IRs: {}".format(-IR / 10))
            print("robobo is at {}".format(current_position))
            print(wheels)

            # move the robot
            rob.move(wheels[0], wheels[1], STEP_SIZE_MS)

            # collect data
            s_trans = wheels[0] + wheels[1]
            s_rot = abs(wheels[0] - wheels[1]) / (2 * M_MAX)
            v_sens = min(max(-IR) / 6.3, 1)

            ep_data.append([s_trans, s_rot, v_sens])

            # printProgressBar(i, STEP_COUNT, prefix='hi', suffix='bye')

            last_position = current_position
            step_time = time.time() - start_time

        # could save some data here for analytics.
        # could do some visualisations.

        # calculate model fitness
        if learning:
            ep_data = pd.DataFrame(ep_data, columns=('s_trans', 's_rot', 'v_sens'))

            ind_fit = evaluate(ep_data, RECOVERY_TIME)
            ind.fitness.values = (ind_fit,)

            print("this model's fitness (previous): ", ind_fit, ' (', fitnesses[0], ')')
            print("difference of genes: ", np.array(pop[0]) - np.array(ind))
            time.sleep(3)

            # delete the inferior model
            fitnesses.append(ind_fit)
            print(fitnesses)

            del pop[1 - np.argmax(fitnesses)]
            del fitnesses[1 - np.argmax(fitnesses)]

            if save_model:
                best = pd.Series(list(pop[0]))
                best.to_csv(MODEL_PATH)

            ### MODEL EVOLUTION ###
            mutant = toolbox.clone(pop[0])
            ind, = toolbox.mutate(mutant)
            time.sleep(3)
            pop.append(ind)

        ep_time = time.time() - ep_start_time


if __name__ == "__main__":
    main()
