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

code cleanup:
- check more func arg passes
- could make individual type a dict
"""

from __future__ import print_function

import time
from datetime import datetime
import random
import array
import sys
import robobo
import signal
import numpy as np
import pandas as pd
from deap import base
from deap import creator
from deap import tools

from neural_network import init_nn_EC
from utils import print_welcome, print_cycle, print_ui, print_ep_ui, save_data_at, generate_name
np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.2f}'.format})


def start_sim(rob):
    signal.signal(signal.SIGINT, terminate_program)
    rob.play_simulation()


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program\n\n")
    sys.exit(1)


def init_deap(sens_names, C, min_value, max_value, min_strategy, max_strategy, **kwargs):
    """
    Function for initializing DEAP. Read the docs before fiddling (google 'DEAP')
    """

    # Register the fitness function and individual types
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode="d",
                   fitness=creator.FitnessMax, strategy=None)
    creator.create("Strategy", array.array, typecode="d")

    # Register two variations of an individual, ES vector and sensor bitstring
    toolbox = base.Toolbox()
    toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
                     (2 * len(sens_names) + 2), min_value, max_value, min_strategy, max_strategy)

    toolbox.register("attr_int", random.randint, 0, 1)
    toolbox.register("bitstring", tools.initRepeat, creator.Individual,
                     toolbox.attr_int, len(sens_names))

    # Register mutation procedures
    # toolbox.register("mutate_bitstring", tools.)
    toolbox.register("mutate_ES", tools.mutESLogNormal, c=C)
    toolbox.decorate("mutate_ES", checkStrategy(min_strategy))

    return toolbox


def generateES(icls, scls, size, imin, imax, smin, smax, MODEL_PATH=False, init_bias=False):
    """Generate neural network weights ('individuals') and evolution parameters ('strategies')"""
    if not MODEL_PATH:

        if not init_bias:
            print("Initializing random model with random biases.")
            ind = icls(random.uniform(imin, imax) for _ in range(size))
        else:
            print("Initializing random model with bias weights [1. 1.].")
            ind = icls([random.uniform(imin, imax) for _ in range(size - 2)] +
                       [1. for _ in range(2)])

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


def take_step(rob, model, params):

    time.sleep(0.1)
    IR = -abs(np.log(np.array(rob.read_irs())))
    IR[np.isinf(IR)] = 0
    current_position = np.array(rob.position())

    wheels = model.predict(np.expand_dims(IR, axis=0))[0] * params['m_max']

    rob.move(wheels[0], wheels[1], params['step_size_ms'])

    step_data = {
        "s_trans": wheels[0] + wheels[1],
        "s_rot": abs(wheels[0] - wheels[1]) / (2 * params['m_max']),
        "v_sens": min(max(-IR) / params['max_sens'], 1),
        "v_total": abs(sum(IR)),
        "IR": IR,
        "wheels": wheels,
        "position": current_position
    }

    return step_data


def evaluate(data, recovery_time):
    """Fitness as defined in Eiben et al. 2015"""
    return (data['s_trans'] * (1 - data['s_rot']) * (1 - data['v_sens']))[recovery_time:].sum()


def learn(toolbox, ep_data, ep, reeval, ind, pop, fitnesses,
          recovery_time, mut_prob_base, mut_prob_0, reeval_rate, **kwargs):
    """Could restructure code in this function... The if statements below are pure horror logic, but I will fix
    them at a later date. `reeval` might also be a nice global, that way it might not need to be called outside
    learn()."""

    # model_dist = 0
    # fitness_winner, fitness_loser = None, None

    mut_rate = (1 - mut_prob_base) * (mut_prob_0 ** ep) + mut_prob_base  # could be nicer
    ep_data = pd.DataFrame(ep_data, columns=ep_data[0].keys())

    # Calculate the fitness and print some stats
    ind_fit = round(evaluate(ep_data, recovery_time), 3)
    ind.fitness.values = (ind_fit,)
    fitnesses.append(ind_fit)

    model_dist = round(((np.array(pop[0]) - np.array(ind)) ** 2).sum(), 3)

    print_ep_ui(ind_fit, fitnesses[0], model_dist, mut_rate, reeval)

    set_reeval = random.random() < reeval_rate

    print('\n\ndebug information:\n')
    print('reeval: ', reeval)
    print('prev_ind: ', pop[0])
    print('ind: ', pop[1])
    print('#pop: ', len(pop))
    print('fit: ', fitnesses)
    time.sleep(10)
    print('setting reeval to True...') if set_reeval else print('Performing selection/recalculation and mutation...')

    if reeval:  # A

        ind_fit = reeval_rate * fitnesses[1] + (1 - reeval_rate) * fitnesses[0]
        ind.fitness.values = (ind_fit,)

        del pop[0]
        fitnesses = [ind_fit]
        fitness_loser = np.nan

        reeval = False

        ind = mutate(toolbox, pop[0], mut_rate)  # C
        pop.append(ind)

        print('\nrecalculated fitness, testing mutant: \n')
        print(ind)

    else:  # B

        fitness_loser = fitnesses[1 - np.argmax(fitnesses)]

        del pop[1 - np.argmax(fitnesses)]
        del fitnesses[1 - np.argmax(fitnesses)]

        reeval = True if set_reeval else False

        if reeval:  # D
            ind = toolbox.clone(ind)
            pop.append(ind)
            print('\nPerforming re-evaluation of : \n')
            print(ind)

        else:  # C
            ind = mutate(toolbox, pop[0], mut_rate)
            pop.append(ind)
            print('\neliminated current best, testing mutant: \n')
            print(ind)

    return ind, pop, reeval, model_dist, fitnesses, fitness_loser


def mutate(toolbox, ind, mut_rate):
    offspring = toolbox.clone(ind)
    ind, = toolbox.mutate_ES(offspring, indpb=mut_rate)
    return ind


def save_model(MODEL_PATH, model_ind, ep, SIM_NUMBER):
    model = pd.Series(list(model_ind))
    model.to_csv(generate_name(model, MODEL_PATH, ep, SIM_NUMBER), index=False)


def append_data(data, ind, ep_data, ep_time, model_dist, fitness_winner=None, fitness_loser=None):
    row = list(ind) + \
          list(pd.DataFrame(ep_data).mean(axis=0)) + \
          [ep_time] + [datetime.today()] + [model_dist] + \
          [fitness_winner, fitness_loser]

    data.append(row)


def main():

    # Initialization
    MODEL_PATH = "src/models/"
    DATA_PATH = "src/data/"
    SIM_NUMBER = 0  # [0,1,2] -> box, pillars, maze
    LEARNING = True

    params = {
        'hardware': False,
        'load_model': False,
        'save_model': True if LEARNING else False,
        'save_data': True,

        'sens_names': ["IR" + str(i + 1) for i in range(8)],
        'ep_count': 1000,
        'step_count': 10,
        'step_size_ms': 200,
        'C': 2.0,
        'min_value': -1.,
        'max_value':  1.,
        'min_strategy': -1.,
        'max_strategy': 1.,
        'mut_prob_0': 0.92,
        'mut_prob_base': 0.3,
        'm_max': 20,
        'recovery_time': 5,
        'init_bias': True,
        'reeval_rate': 0.2 if LEARNING else 0,
        'max_sens': 0.65
    }

    if params['hardware']:
        rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7")
    else:
        rob = robobo.SimulationRobobo(number=["","#2","#0"][SIM_NUMBER]).connect(address='172.20.10.2', port=19997)
        # rob2 = robobo.SimulationRobobo(number=["","#2","#0"][1]).connect(address='172.20.10.3', port=19998)

    print_welcome()
    toolbox = init_deap(**params)

    if params['load_model']:
        ind = toolbox.individual(MODEL_PATH=MODEL_PATH)
    else:
        ind = toolbox.individual(init_bias=params['init_bias'])

    pop = [ind, ind]
    fitnesses = [-10000]

    data = []
    reeval = False


    ########## MAIN ALGORITHM ##########

    start_time = datetime.now()
    for ep in range(params['ep_count']):

        print_cycle(ep)
        ep_start_time = time.time()

        # start_sim(rob)
        signal.signal(signal.SIGINT, terminate_program)
        rob.play_simulation()

        model = init_nn_EC(input_dims=len(params['sens_names']), output_dims=2, ind=ind, dropout=LEARNING)


        ########## EVALUATION ##########

        print('Testing these model parameters: \n', np.array(model.get_weights()).T)

        ep_data = []

        for i in range(params['step_count']):

            step_data = take_step(rob, model, params)
            ep_data.append(step_data)

            print_ui(step_data['IR'], step_data['position'], step_data['wheels'],
                     model, start_time, ep, i, params['step_count'])

            time.sleep(1)  # if on slow computer


        ########## EVOLUTION ##########

        model_dist = 0  # if not learning
        fitness_loser = 0

        if LEARNING:
            rob.move(0, 0, 1)
            print('fitnesses in: ', fitnesses)
            ind, pop, reeval, model_dist, fitnesses, fitness_loser = learn(toolbox, ep_data, ep, reeval,
                                                                           ind, pop, fitnesses, **params)
            print('fitnesses out: ', fitnesses)
            
            if params['save_model']:
                save_model(MODEL_PATH, pop[0], ep, SIM_NUMBER)

            # fitnesses = [fitness_winner]

        # save the episode data
        ep_time = time.time() - ep_start_time
        append_data(data, pop[0], ep_data, ep_time, model_dist, fitnesses[0], fitness_loser)

        # write data to .csv
        save_data_at(DATA_PATH, data, ep) if params['save_data'] else None


if __name__ == "__main__":
    main()
