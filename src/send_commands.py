"""
Current approach:
Q-L with NN
...

todo's
- Review more literature and previous work on how to solve many of these problems...
- adapting reward structure
    > based on Eiben et al. but differs and is a lot more improvised / sloppy. Could be more 'rigorous'
    > termination would ideally be one step earlier... or at least reward assignment. Could implement a negative
      reward that sends back to previous states?
    > Eligibility traces could be useful for sending rewards further back? Credit assignment problem
    > in ML terms, this boils down to improving the quality of the data. The further back the reward can be associated
      with specific actions, the earlier the robot will know where to go. Eligibility traces with lambda = 3 would
      seem sensible, as the robot usually detects a wall about 3 steps from crashing into it
- Improving neural network
    > loss could (should?) be redefined as squared distance of the changed Q-value only
- Experiments: Time alive over episodes of the algorithm?
- Fun / easy to test: different types of action configurations.
    > could encode action step size (STEP_SIZE_MS) as parameter in action? Then the robot can learn for itself when to
      drive longer for a big reward or be careful.
    > More actions? smaller turn circles, etc.
- Implementing camera data -> extract some metrics / statistics / convNet features (whatever you can think of)
    > Feed data into the neural network, see if it helps
    > not too many extra nodes... or weigh them very low. Else learning will take too long
- Can the algorithm be rearranged / whatever to have more continuous driving without messing up the Q-algorithm?
  Must be possible for the next action to be selected while driving... i.e. select action off the previous state?
    > Improving computation speed?
    > threading to allow for simultaneous action decisionmaking and driving?
- look into the signal numbers to figure out how to give commands while script is running
- reorganize script (put stuff into functions, split up files, etc.)
- many of the parameters can probably be set relative to each other. Forget rate depends on episode length, etc...

- issues with timing: S and S_prime might not be taken at the right time.

Notes
- keeping amount of episodes low is important in the interest of time
- Experience replay might be very helpful? or some other way to send a negative reward further back - agent seems
to steer into walls too often when it should really know to veer off correctly.

"""


#!/usr/bin/env python2
from __future__ import print_function

import vrep
from data_structure import Data, EpisodeData

import time
import random
import math as m

import numpy as np
np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.2f}'.format})
import pandas as pd
from data_structure import Data, EpisodeData
from neural_network import init_nn, save_nn

import robobo
# import cv2
import sys
import signal
# import prey
from vrep import VrepApiError


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program\n\n")
    sys.exit(1)


def start_simulation(rob):
    try:
        time.sleep(0.1)
        rob.play_simulation()

    except:
        print("Simulation startup failed. Retrying...")
        time.sleep(1)
        start_simulation(rob)

    print("starting at position {}".format(rob.position()))


def e_greedy_action(Q, A, e=0.1):
    action = e_greedy(Q, e)
    wheel = A[action]
    return action, {'left': wheel[0], 'right': wheel[1]}


def e_greedy(Q, e=0.1):
    if random.random() < e:
        return random.randint(0,len(Q)-1)
    else:
        return np.random.choice(np.where(Q == Q.max())[0])


def main():

    global crashed

    EP_COUNT = 100
    STEP_COUNT = 50
    STEP_SIZE_MS = 500
    CRASH_SENSOR_BOUNDARY = -0.43  # negative for simulation, positive for RW. Not used right now
    CRASH_POSITION_BOUNDARY = 0.02

    A = [(20,20), (0,20), (20,0)]#, (10,5), (5,10)]  # (-25,-25),
    epsilon = 0.1
    epsilon_decaying = 0.5
    gamma = 0.1
    recency_factor = 0.01  # higher parameter setting -> forget older information faster
    # proximity_factor = 1  # how heavily to penalize proximity to obstacle

    ACTION_NAMES = ['forward', 'sharp left', 'sharp right']  # 'left', 'right']  # 'backward',
    SENS_NAMES = ["IR" + str(i + 1) for i in range(8)]

    # Initialize the robot -> SET THESE PARAMETERS!
    hardware = False
    # nn_from_file = True if input('Would you like to use a pre-trained neural network? (y/n') == 'y' else False
    nn_from_file = True
    learning = False  # Disable this if you want to run a pre-trained network
    if nn_from_file is True:
        print('loading network, disabling learning...')
        learning = False

    if hardware:
        rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7")
    else:
        rob = robobo.SimulationRobobo().connect(address='172.20.10.3', port=19997)
        rob.stop_world()
        time.sleep(0.1)

    if not learning:
        epsilon = 0
        epsilon_decaying = 0

    # Initialize the data structure and neural network
    eps = []
    hidden_layers = [16, 12]
    model = init_nn(input_dims=len(SENS_NAMES), output_dims=len(A),
                    hidden_layers=hidden_layers, from_file=nn_from_file)

    for episode in range(EP_COUNT):

        data = EpisodeData(ACTION_NAMES, sens_names=SENS_NAMES)

        signal.signal(signal.SIGINT, terminate_program)
        start_simulation(rob)

        ### INITIALIZATION ###
        print('\n--- episode {} ---'.format(episode + 1))

        S = np.log(np.array(rob.read_irs()))
        S[np.isinf(S)] = 0
        last_position = np.array([0,0,0])

        ########## Q-LEARNING LOOP ##########

        crashed = False

        for i in range(STEP_COUNT):
            start_time = time.time()

            print('\n--- step {} ---'.format(i+1))

            # pos = rob.position()

            ### ACTION SELECTION & EXECUTION ###
            Q_s = model.predict(np.expand_dims(S, 0))[0]

            # request wheel speed parameters for max action
            action, wheels = e_greedy_action(Q_s, A, epsilon)# epsilon_decaying ** (1 + (0.1 * episode)))

            # move the robot
            rob.move(wheels['left'], wheels['right'], STEP_SIZE_MS)

            if learning:
                time.sleep(STEP_SIZE_MS/1000)

            ### OBSERVING NEXT STATE ###
            S_prime = np.log(np.array(rob.read_irs()))
            S_prime[np.isinf(S_prime)] = 0

            print("ROB IRs: {}".format(S_prime / 10))
            current_position = np.array(rob.position())
            print("robobo is at {}".format(current_position))

            # observe the reward
            s_trans = wheels['left'] + wheels['right']  # translational speed of the robot
            s_rot = abs(wheels['left'] - wheels['right'])  # rotational speed of the robot

            if hardware:
                crashed = False
                raise NotImplementedError("Haven't implemented this, I suggest using a threshold for the sensor (see"
                                          "code below this statement)")
            else:
                dist = np.linalg.norm(last_position - current_position)
                crashed = min(S_prime[3:] / 10) < CRASH_SENSOR_BOUNDARY or dist < CRASH_POSITION_BOUNDARY

            if not crashed:
                # reward = 1 + min(S) * proximity_factor  # - (wheels == A[1])
                # see Eiben et al. for this formula
                reward = s_trans * (1 - 0.9 * (s_rot / 20)) * (1 - (min(S_prime[3:]) / -0.65))
            else:
                reward = -400

            # Retrieve Q values from neural network
            Q_prime = model.predict(np.expand_dims(S_prime, 0))[0]

            ### LEARNING ###

            Q_target = reward + (gamma * np.argmax(Q_prime))

            Q_targets = np.copy(Q_s)
            Q_targets[action] = Q_target

            ### SAVE DATA ###

            # pos = np.array([1,2,3])
            data.update(i, S, Q_s, Q_targets, reward)  # pos removed

            ### TERMINATION CONDITION ###

            # if S == S_prime and not S.sum() == 0:  # np.isinf(S).any() is False:
            #     print('Termination condition reached')
            #     break

            S = np.copy(S_prime)
            last_position = current_position

            print("crashed: ", crashed)
            print("chosen action:", ACTION_NAMES[action])
            print('reward: ', reward)
            print("Q_s (NN output): ", Q_s)
            print("Updated Q-values: " + str(Q_targets))

            elapsed_time = time.time() - start_time

            if crashed:
                break

        # terminate the episode data and store it
        data.terminate()
        eps.append(data)

        if learning:

            print("\n----- Learning ----\n")
            X = pd.concat([ep_data.sens for ep_data in eps])
            y = pd.concat([ep_data.Q_targets for ep_data in eps])

            # # calculate sample weights
            # ep_lengths = [len(ep_data.sens) for ep_data in eps[::-1]]
            # sample_weights = []
            #
            # for i, ep_length in enumerate(ep_lengths):
            #     sample_weights = sample_weights + ([(1 - recency_factor) ** i] * ep_length)

            # perform learning over the episode
            model.fit(X, y, epochs=100) #sample_weight=np.array(sample_weights))

        # # perform an evaluation of the episode (probably not necessary till later)
        # model.evaluate(data)

        # time.sleep(0.1)

        # pause the simulation and read the collected food
        # rob.pause_simulation()

        rob.sleep(1)

        if crashed or episode == EP_COUNT:
            save_nn(model)
            print('Robot crashed, resetting simulation...')
            data.terminate(crashed)
            # could implement something here to save the experience if resetting the simulation!

        if crashed:
            rob.stop_world()

    rob.stop_world()

## Test script ###

# def main():
#     signal.signal(signal.SIGINT, terminate_program)
#
#     rob = robobo.HardwareRobobo(camera=True).connect(address="172.20.10.10")
#     # rob = robobo.SimulationRobobo().connect(address='172.20.10.10', port=19997)
#
#     rob.play_simulation()
#
#     # Following code moves the robot
#     for i in range(20):
#         print("robobo is at {}".format(rob.position()))
#         print("ROB Irs: {}".format(np.log(np.array(rob.read_irs())) / 10))
#         rob.move(5, 5, 2000)
#
#     print("robobo is at {}".format(rob.position()))
#     rob.sleep(1)
#
#     # Following code moves the phone stand
#     rob.set_phone_pan(343, 100)
#     rob.set_phone_tilt(109, 100)
#     time.sleep(1)
#     rob.set_phone_pan(11, 100)
#     rob.set_phone_tilt(26, 100)
#
#     # Following code makes the robot talk and be emotional
#     rob.set_emotion('happy')
#     rob.talk('Hi, my name is Robobo')
#     rob.sleep(1)
#     rob.set_emotion('sad')
#
#     # Following code gets an image from the camera
#     image = rob.get_image_front()
#     cv2.imwrite("test_pictures.png", image)
#
#     time.sleep(0.1)
#
#     # IR reading
#     for i in range(1000000):
#         print("ROB Irs: {}".format(np.log(np.array(rob.read_irs())) / 10))
#         time.sleep(0.1)
#
#     # pause the simulation and read the collected food
#     rob.pause_simulation()
#
#     # Stopping the simualtion resets the environment
#     rob.stop_world()


if __name__ == "__main__":
    main()
