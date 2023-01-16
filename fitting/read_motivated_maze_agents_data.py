from operator import index

import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime

from fitting.med2rawconverter import load_and_parse_med_file

data_path = '/Users/gkour/repositories/plusmaze/fitting/motivation_behavioral_data_raw/'
stages = ['odor1_WR', 'odor2_WR', 'odor2_XFR', 'odor3_WR', 'spatial_WR']


def get_all_data_from_csv_for_agent(experiment_num, agent_num):
    result_table_concat = []
    experiment_dir = get_experiment_dir(experiment_num)
    for stage_index, stage in enumerate(stages):
        files_for_agent_in_stage = get_files_for_agent_in_stage_sorted(
            experiment_dir, stage, agent_num)
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


def get_files_for_agent_in_stage_sorted(experiment_dir, stage, agent_num):
    files_for_agent_in_stage = []
    for file in glob(experiment_dir + '/' + stage + '/*'):
        if file.endswith(f"{agent_num}"):
            files_for_agent_in_stage.append(file)
    return sort_files_by_date(files_for_agent_in_stage)


def get_date_from_file(file):
    return file.split('/')[-1][1: 11]


def sort_files_by_date(files):
    files.sort(key=lambda file: datetime.strptime(
        get_date_from_file(file), '%Y-%m-%d'))
    return files


def get_data_from_csv(file, stage_index, day_index):
    df = load_and_parse_med_file(file)
    #df = pd.read_csv(file)
    df = add_stage_index_and_day_to_df(df, stage_index, day_index)
    df = parse_table_stage_by_odor(df)
    return df


def add_stage_index_and_day_to_df(df, stage_index, day_index):
    df['stage'] = stage_index + 1
    df['day in stage'] = day_index + 1
    return df


def parse_table_stage_by_odor(df):
    df = set_odor_and_color_column(df)
    df['action'] = df.apply(lambda row: row.chosen_arm, axis=1)
    df['reward'] = df.apply(lambda row: 1 if row.trial_outcome == 1 else 0 if row.trial_outcome == 2 else None, axis=1)
    df['reward_type'] = df.apply(lambda row: int(reward_type(row)), axis=1)
    return df[['stage', 'day in stage', 'trial', 'A1o', 'A1c', 'A2o', 'A2c', 'A3o', 'A3c', 'A4o', 'A4c', 'action', 'reward', 'reward_type']]


def set_odor_and_color_column(df):

    # Possible cues combinations:
    # (1, 0)(0, 1)(1, 0)(0, 1)
    # (1, 0)(0, 1)(0, 1)(1, 0)
    # (1, 1)(0, 0)(1, 1)(0, 0)
    # (1, 1)(0, 0)(0, 0)(1, 1)

    # (0, 1)(0, 1)(0, 1)(0, 1)
    # (0, 1)(0, 1)(1, 0)(1, 0)
    # (0, 0)(1, 1)(0, 0)(1, 1)
    # (0, 0)(1, 1)(1, 1)(0, 0)

    def cues_combination (correct_p1, correct_p2, configuration):
        cues = [[-1,-1],[-1,-1],[-1,-1],[-1,-1]]

        cues[0][0] = 1 if correct_p1 == 1 else 0 # handle odor in door 1
        cues[1][0] = 1-cues[0][0]               # handle odor in door 2

        cues[2][0] = 1 if correct_p2 == 3 else 0 #handle odor in 3
        cues[3][0] = 1 - cues[2][0]                #handle odor in 4

        cues[0][1] = cues[0][0] if configuration==1 else 1-cues[0][0]
        cues[1][1] = cues[1][0] if configuration==1 else 1-cues[1][0]

        # take the same combination from the first part of the maze according to the relevant cue.
        cues[2][1] = cues[0][1] if cues[2][0]==cues[0][0] else cues[1][1]
        cues[3][1] = 1 - cues[2][1]

        return cues

    df['combination'] = df.apply(lambda row: cues_combination(int(row.correct_p1),int(row.correct_p2), int(row.configuration)), axis=1)

    df['A1o'] = df.apply(lambda row: row.combination[0][0], axis=1)
    df['A1c'] = df.apply(lambda row: row.combination[0][1], axis=1)

    df['A2o'] = df.apply(lambda row: row.combination[1][0], axis=1)
    df['A2c'] = df.apply(lambda row: row.combination[1][1], axis=1)

    df['A3o'] = df.apply(lambda row: row.combination[2][0], axis=1)
    df['A3c'] = df.apply(lambda row: row.combination[2][1], axis=1)

    df['A4o'] = df.apply(lambda row: row.combination[3][0], axis=1)
    df['A4c'] = df.apply(lambda row: row.combination[3][1], axis=1)

    df.drop('combination', inplace=True, axis=1)
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


def export_motivational_experiment_data():
    expr_data = {1: [1, 2], 2: [1], 4: [6, 7, 8], 5: [1, 2], 6: [10, 11]}

    for expr in expr_data:
        for rat in expr_data[expr]:
            save_pd_to_csv(get_all_data_from_csv_for_agent(expr,rat), './fitting/motivation_behavioral_data/output_expr{}_rat{}.csv'.format(expr, rat))


if __name__ == '__main__':

    export_motivational_experiment_data()