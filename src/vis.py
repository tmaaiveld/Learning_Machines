"""
data is a 2-D dataframe. Rows (axis 0) represent episodes. Columns (axis 1) represent variables.
I saved all genes (neural network weights) as 'weight_i_j' where (i,j) are input and output nodes respectively,
and bias weights are labeled 'bias_1' and 'bias_2'.

I also saved the episode time (episode_time_ms), time of day when saving the episode data (time_of_day),
episode length (ep_length) because not all episodes are equally long during training, distance_to_previous_model and
fitness.

A decent start would be plotting episode time over reward.
Distance to previous model tells you something about how strongly the algorithm is exploring. Mutation is adaptive,
so continually evaluating bad fitnesses increases the mutation step size (exploration parameter).
"""

from os import listdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


def create_plot(y, x):
    pass


def get_session_data(session, ):
    """Takes a list of episode datasets as input and returns datasets associated with a session."""
    pass


DATA_PATH = "data/"
data_files = [DATA_PATH + filename for filename in listdir(DATA_PATH)]
datasets = []

for file in data_files:
    data = pd.read_csv(file)
    datasets.append(data)
    print(file)
    print(len(data))

quit()

dataset_sizes = [len(dataset) for dataset in datasets]
data = datasets[np.argmax(dataset_sizes)]

print(data.head(10))

ax = sns.lineplot(x=data.index, y="fitness", data=data)
plt.show()

# compute the sum of the episode times to find out how long the script actually takes to run...

# data = pd.read_csv(DATA_PATH)

# print(data.head(10))