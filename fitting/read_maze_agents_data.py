import os
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
import scipy


def get_all_data_from_mat_for_agent(rat_data_dir):
    result_table_concat = []
    for stage_index, stage in enumerate(stages):
        files_for_agent_in_stage = get_files_for_agent_in_stage_sorted(rat_data_dir, stage)
        day = 0
        last_date = get_date_from_file(files_for_agent_in_stage[0])
        for file in files_for_agent_in_stage:
            curr_date = get_date_from_file(file)
            if last_date != curr_date:
                last_date = curr_date
                day = day + 1
            result_table_concat.append(
                get_data_from_csv(file, stage_index, day))
    return pd.concat(result_table_concat)


def get_experiment_dir(experiment_num):
    for directory in glob(data_path + '*'):
        if f"experiment{experiment_num}" in directory:
            return directory


def get_files_for_agent_in_stage_sorted(rat_dir, stage):
    files_for_agent_in_stage = []
    for file in glob(rat_dir + '/' + stage + '/*'):
        files_for_agent_in_stage.append(file)
    return sort_files_by_date(files_for_agent_in_stage)


def get_date_from_file(file):
    return file.split('/')[-1].split('.')[0].split('_')[1]


def sort_files_by_date(files):
    files.sort(key=lambda file: datetime.strptime(get_date_from_file(file), '%d%m'))
    return files


def get_data_from_csv(file, stage_index, day_index):
    #trial, r, w, c, np, cup, ic_1, ic_2
    df = pd.DataFrame(scipy.io.loadmat(file)['allTrials'],
                      columns=['trial', 'door_relevant_cue1', 'door_relevant_cue2', 'chosen_arm','np_time', 'cup_time',
                               'door_irrelevant_cue1', 'door_irrelevant_cue2'])
    df = add_stage_index_and_day_to_df(df, stage_index, day_index)
    df = parse_table_stage_by_odor(df, stages[stage_index])
    return df


def add_stage_index_and_day_to_df(df, stage_index, day_index):
    df['stage'] = stage_index + 1
    df['day in stage'] = day_index + 1
    return df


def parse_table_stage_by_odor(df, stage):
    df = set_odor_and_color_column(df, stage)
    df['action'] = df['chosen_arm']
    df['reward'] = df.apply(lambda row: 1 if row.chosen_arm == row.door_relevant_cue1 else (None if np.isnan(row.chosen_arm) else 0), axis=1)
    return df[['stage', 'day in stage', 'trial', 'A1o', 'A1c', 'A2o', 'A2c', 'A3o', 'A3c', 'A4o', 'A4c', 'action', 'reward']]


def set_odor_and_color_column(df, stage):
    if 'ODOR' in stage:
        df['door_odor_cue1'] = df['door_relevant_cue1']
        df['door_odor_cue2'] = df['door_relevant_cue2']
        df['door_color_cue1'] = df['door_irrelevant_cue1']
        df['door_color_cue2'] = df['door_irrelevant_cue2']
    else:
        df['door_odor_cue1'] = df['door_irrelevant_cue1']
        df['door_odor_cue2'] = df['door_irrelevant_cue2']
        df['door_color_cue1'] = df['door_relevant_cue1']
        df['door_color_cue2'] = df['door_relevant_cue2']

    for arm in range(1,5):
        df['A{}o'.format(arm)] = df.apply(
            lambda row: 1 if row.door_odor_cue1 == arm else 0 if row.door_odor_cue2 == arm else -1, axis=1)
        df['A{}c'.format(arm)] = df.apply(
            lambda row: 1 if row.door_color_cue1 == arm else 0 if row.door_color_cue2 == arm else -1, axis=1)
    return df


def reward_type(row):
    if row.action in [1, 2]:
        return 1  # food
    elif row.action in [3, 4]:
        return 2  # water
    else:
        return 0


def save_pd_to_csv(df, file_name):
    df.to_csv(file_name, index=False)


def export_maze_experiment_data():
    rats = [0,1,2,3,4,5,6,7,8]

    global data_path
    global stages
    data_path = 'fitting/maze_behavioral_data_raw'
    stages = ['ODOR1', 'ODOR2', 'LEDs1']

    for rat in rats:
        save_pd_to_csv(get_all_data_from_mat_for_agent(os.path.join(data_path, "rat{}".format(rat))),
                       './fitting/maze_behavioral_data/output_expr_rat{}.csv'.format(rat))


def export_maze_led_first_experiment_data():
    rats = [33,34,35,36,37,38,39,40,41,42]

    global data_path
    global stages
    data_path = 'fitting/maze_behavioral_led_first_data_raw'
    stages = ['LEDs1', 'ODOR1']

    for rat in rats:
        save_pd_to_csv(get_all_data_from_mat_for_agent(os.path.join(data_path, "rat{}".format(rat))),
                       './fitting/maze_behavioral_led_first_data/output_expr_rat{}.csv'.format(rat))


if __name__ == '__main__':
    export_maze_led_first_experiment_data()