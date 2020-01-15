import sys
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from os import listdir


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    print('\n')

    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '#' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

    print('\n')


def print_welcome():
    print("\n\n\n     ###########################################     " +
              "\n     ########   WELCOME TO FOXTROT-ES   ########     " +
              "\n     ###########################################     \n\n\n")
    time.sleep(1)


def print_cycle(ep, i=False):
    if not i:
        time.sleep(0.001)
        print('\n--- episode {} ---\n'.format(ep + 1))
    else:
        print('\n--- step {} of episode {} ---\n'.format(i + 1, ep + 1))


def print_rob(IR, current_position, wheels):
    print("ROB IRs: {}".format(IR / 10))
    print("robobo is at {}".format(current_position))
    print("Wheel speeds: [{:.2f} {:.2f}]".format(wheels[0], wheels[1]))


def print_time(td):
    t = td.seconds
    hours, minutes, seconds = t // 3600, t % 3600 / 60, t % 60
    print("Elapsed time: {:02d}:{:02d}:{:02d}".format(hours, minutes, seconds))


def print_ui(IR, current_position, wheels, model, start_time, ep, i, max_step):
    print_cycle(ep, i)
    print('currently testing: \n')
    print(np.array(model.get_weights()).T)
    print('\n')
    current_time = datetime.now() - start_time
    print_time(current_time)
    print_rob(IR, current_position, wheels)
    print_progress(i, max_step, bar_length=50)


def print_ep_ui(ind_fit, prev_fit, model_dist, mut_rate, reeval):
    print('\n\n')
    print("Performing re-evaluation...") if reeval else None
    print("this model's fitness (previous): ", ind_fit, ' (', prev_fit, ')')
    print("Euclidian distance to previous model: ", model_dist)
    print('current mutation rate: ', mut_rate)


def save_data_at(data_path, data, episode):

    data_columns = ['weight_1_' + str(i + 1) for i in range(8)] + \
                   ['weight_2_' + str(i + 1) for i in range(8)] + \
                   ['bias_' + str(i + 1) for i in range(2)] +     \
                   ['avg_s_trans', 'avg_s_rot', 'avg_v_sens', 'avg_v_tot'] +   \
                   ['episode_time_ms', 'time_of_day', 'distance_to_previous_model'] + \
                   ['fitness', 'loser_fitness']

    data = pd.DataFrame(data, columns=data_columns)
    today = str(datetime.today().day)

    sessions_today = [int(f.split("_")[1]) for f in listdir(data_path) if f.split("_")[2] == today]

    if episode == 0:
        model_index = max(sessions_today) + 1 if sessions_today else 0
    else:
        model_index = max(sessions_today) if sessions_today else 0

    data.to_csv(data_path + "data_" + str(model_index) + "_" + today + "_jan.csv")


def generate_name(model, model_path, episode, sim_number):

    prev_model_number = [f.split('_')[1] for f in listdir(model_path)][-1] if listdir(model_path) else -1

    if episode == 0:
        model_number = int(prev_model_number) + 1
    else:
        model_number = prev_model_number

    model_name = "model_" + str(model_number) +  "_sim" + str(sim_number)

    model.to_csv(model_path + model_name, index=False)
