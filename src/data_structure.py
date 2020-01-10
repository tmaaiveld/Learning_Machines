"""
Data structure file for send_commands.py. Needs some work, probably better to put it all
in one dataframe.
"""

import numpy as np
import pandas as pd


class Data:
    """

    """
    def __init__(self):
        self.episodes = []

    def add_episode(self, episode_data):
        self.episodes.append(episode_data)


class EpisodeData:
    """
    Data for a single episode.

    :var sens_names: names of the sensor
    :var self.sens: a Pandas DataFrame containing eight IR readings and translational/rotational speed
    :var self.win: Boolean representing whether the agent reached the goal state or not
    :var self.pos: x, y, z coordinates of the agent (simulation only)

    move this into a separate file later.
    """

    def __init__(self, actions, sens_names, win_coordinates=0):
        self.sens = pd.DataFrame(columns=sens_names)
        self.pos = pd.DataFrame(columns=['x', 'y', 'z'])
        self.reward = pd.Series()
        
        self.Q = pd.DataFrame(columns=actions)
        self.Q_targets = pd.DataFrame(columns=actions)
        self.wheels = pd.DataFrame(columns=['left', 'right'])
        self.action = pd.Series()

        self.win = False
        self.win_coordinates = win_coordinates

    def update(self, i, S, step_Q, step_Q_targets, step_pos, reward):
        """
        :param step_sens: IR sensor readings at a given time step.
        :param step_pos: simulation position readings
        :return: updated parameters.
        """

        self.sens.loc[i] = S
        # self.pos.loc[i] = S[-3:]
        self.Q.loc[i] = step_Q
        self.Q_targets.loc[i] = step_Q_targets
        self.pos.loc[i] = step_pos
        self.reward.loc[i] = reward

    def terminate(self):
        self.win = (self.win_coordinates in self.pos)  # probably won't work yet
        self.action = self.Q.argmax(axis=1)
