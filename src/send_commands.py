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
"""


#!/usr/bin/env python2
from __future__ import print_function
from data_structure import Data, EpisodeData

import time
import random
import math as m

import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense
from data_structure import Data, EpisodeData
from model import Model

import robobo
# import cv2
import sys
import signal
import prey


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


def e_greedy_action(Q, A, e=0.1):
    if random.random() < e:
        action = np.argmax(Q)
    else:
        action = random.randint(0, len(A)-1)

    wheel = A[action]

    return {'left': wheel[0], 'right': wheel[1]}


def out_of_bounds(robot_position):
    """Could also use time based termination?"""
    raise NotImplementedError()


def main():

    EP_COUNT = 1
    STEP_COUNT = 10
    STEP_SIZE_MS = 2000
    SUCCESS_COORDINATES = (1,1,1)
    A = [(5,5), (-5,-5), (0,5), (5,0)]
    SENS_NAMES = ["s_trans", "s_rot"] + ["IR" + str(i + 1) for i in range(8)]
    gamma = 0.1
    proximity_factor = 1  # how heavily to penalize proximity to obstacle

    # Initialize the data structure and neural network [OOP?]
    eps = []
    model = Model()  # replace with Keras model

    for episode in range(EP_COUNT):
        data = EpisodeData(A, sens_names=SENS_NAMES)

        signal.signal(signal.SIGINT, terminate_program)

        # rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7")
        rob = robobo.SimulationRobobo().connect(address='192.168.1.73', port=19997)

        rob.play_simulation()

        ### INITIALIZATION ###

        speed = np.array([0,0])
        ir = np.log(np.array(rob.read_irs()))  # replace -infs with 0
        pos = rob.position()
        print("starting at position {}".format(rob.position()))

        S = np.concatenate([speed, ir, pos])

        print(S)

        ########## Q-LEARNING LOOP ##########

        for i in range(STEP_COUNT):
            start_time = time.time()

            ### ACTION SELECTION & EXECUTION ###

            Q_s = model.predict(S)
            # request wheel speed parameters for max action
            action = e_greedy_action(Q_s, A)

            # move the robot
            print(action['left'])
            rob.move(action['left'], action['right'], STEP_SIZE_MS)

            ### OBSERVING NEXT STATE ###

            s_trans = action['left'] + action['right']
            s_rot = abs(action['left'] - action['right'])
            speed = np.array([s_trans, s_rot])

            ir = np.log(pd.array(rob.read_irs()))
            ir[np.where(ir == np.NINF)] = 0

            pos = rob.position()

            print("ROB IRs: {}".format(ir / 10))
            print("robobo is at {}".format(rob.position()))

            S_prime = np.concatenate([speed, ir, pos])  # extend with wheel speed data?

            # Retrieve Q values from neural network
            Q_prime = model.predict(S)  # needs work

            # observe the reward
            reward = -1 + min(ir) ** proximity_factor  # + add a term for fulfilling the goal

            ### LEARNING ###

            Q_target = reward + (gamma * np.argmax(Q_prime))

            Q_targets = np.copy(Q_s)
            Q_targets[np.argmax(Q_s)] = Q_target

            model.train(S, Q_targets)  # should probably implement / research batch learning!

            ### SAVE DATA ###

            data.update(i, S, Q_s, reward)

            S = S_prime

            elapsed_time = time.time() - start_time
            time.sleep(STEP_SIZE_MS - elapsed_time)

        # terminate the episode data and store it
        data.terminate()
        eps.append(data)

        # perform an evaluation of the episode
        model.evaluate(data)

        rob.sleep(1)

        time.sleep(0.1)

        # pause the simulation and read the collected food
        rob.pause_simulation()

        # Stopping the simulation resets the environment
        rob.stop_world()

        # save the episode data
        eps.append(data)


# example

# def main():
#     signal.signal(signal.SIGINT, terminate_program)
#
#     # rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7")
#     rob = robobo.SimulationRobobo().connect(address='192.168.1.73', port=19997)
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
