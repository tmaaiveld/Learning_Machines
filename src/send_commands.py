"""
Current approach:
Q-L with NN
...

Potential problems with this approach:
- Credit assignment is difficult without intermediary rewards
- RNN infrastructure might make more sense, add new episode data after each episode
- try discretizing sensor readings to reduce state space and improve nn prediction?

To explore:
- Evolve optimal parameters using EA?
- implementing batch learning?
- using only l/r commands?
"""


#!/usr/bin/env python2
from __future__ import print_function
from data_structure import Data, EpisodeData

import time
import random
import math as m

import numpy as np
import pandas as pd
from data_structure import Data, EpisodeData
from neural_network import init_nn

import robobo
# import cv2
import sys
import signal
# import prey


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


def e_greedy_action(Q, A, e=0.1):
    if random.random() > e:
        action = np.argmax(Q)
    else:
        action = random.randint(0, len(A)-1)

    wheel = A[action]

    return {'left': wheel[0], 'right': wheel[1]}


def main():

    EP_COUNT = 10
    STEP_COUNT = 10
    STEP_SIZE_MS = 1000
    A = [(20,20), (-20,-20), (0,20), (20,0)]
    epsilon = 0.1
    gamma = 0.1

    SENS_NAMES = ["IR" + str(i + 1) for i in range(8)]
    proximity_factor = 1  # how heavily to penalize proximity to obstacle

    # Initialize the data structure and neural network
    eps = []
    model = init_nn(8, 4, [16, 12])

    # rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7")
    rob = robobo.SimulationRobobo().connect(address='172.20.10.2', port=19997)

    for episode in range(EP_COUNT):
        data = EpisodeData(A, sens_names=SENS_NAMES)

        signal.signal(signal.SIGINT, terminate_program)
        rob.play_simulation()

        ### INITIALIZATION ###

        S = np.log(np.array(rob.read_irs()))
        S[np.isinf(S)] = 0

        print("starting at position {}".format(rob.position()))
        print(S)

        ########## Q-LEARNING LOOP ##########

        for i in range(STEP_COUNT):
            start_time = time.time()

            print('--- starting cycle {} ---'.format(i+1))

            # pos = rob.position()

            ### ACTION SELECTION & EXECUTION ###
            Q_s = model.predict(np.expand_dims(S, 0))[0]

            # request wheel speed parameters for max action
            action = e_greedy_action(Q_s, A, epsilon)

            # move the robot
            rob.move(action['left'], action['right'], STEP_SIZE_MS)

            ### OBSERVING NEXT STATE ###

            S_prime = np.log(np.array(rob.read_irs()))
            S_prime[np.isinf(S_prime)] = 0

            print("ROB IRs: {}".format(S / 10))
            # print("robobo is at {}".format(rob.position()))

            # observe the reward
            crashed = False  # still needs to be defined.

            print('driving backwards = ', (action == A[1]))

            if not crashed:
                reward = 1 + min(S) * proximity_factor  #- (action == A[1])
            else:
                reward = -100

            print('reward: ', reward)

            # Retrieve Q values from neural network
            Q_prime = model.predict(np.expand_dims(S_prime, 0))[0]

            ### LEARNING ###

            Q_target = reward + (gamma * np.argmax(Q_prime))

            Q_targets = np.copy(Q_s)
            Q_targets[np.argmax(Q_s)] = Q_target

            ### SAVE DATA ###

            # pos = np.array([1,2,3])
            data.update(i, S, Q_s, Q_targets, reward)  # pos removed

            ### TERMINATION CONDITION ###

            # if S == S_prime and not S.sum() == 0:  # np.isinf(S).any() is False:
            #     print('Termination condition reached')
            #     break

            S = np.copy(S_prime)

            print("Q_s (NN output): ", Q_s)
            print("Updated Q-values: " + str(Q_targets))

            elapsed_time = time.time() - start_time
            # time.sleep(STEP_SIZE_MS - elapsed_time)

        # terminate the episode data and store it
        data.terminate()
        eps.append(data)

        # perform learning over the episode
        model.fit(data.sens, data.Q_targets, epochs=3)

        # # perform an evaluation of the episode (probably not necessary till later)
        # model.evaluate(data)

        rob.sleep(1)

        time.sleep(0.1)

        # pause the simulation and read the collected food
        rob.pause_simulation()

        print('finished episode')

    # Stopping the simulation resets the environment
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
