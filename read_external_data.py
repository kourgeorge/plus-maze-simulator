import pandas as pd
from glob import glob
from datetime import datetime

data_path = './MED data/'
stages = ['odor1_WR', 'odor2_WR', 'odor2_XFR', 'odor3_WR', 'spatial_WR']

def get_all_data_from_csv_for_agent(experiment_num, agent_num):
    result_table_concat = []
    experiment_dir = get_experiment_dir(experiment_num)
    for stage_index, stage in enumerate(stages):
        files_for_agent_in_stage = get_files_for_agent_in_stage_sorted(experiment_dir, stage, agent_num)
        for file_index, file in enumerate(files_for_agent_in_stage):
            result_table_concat.append(get_data_from_csv(file, stage_index, file_index)) # need to change this and transform the table to what george asked.
    return pd.concat(result_table_concat)

def get_experiment_dir(experiment_num):
    for directory in glob(data_path + '*'):
        if f"experiment{experiment_num}" in directory:
            return directory

def get_files_for_agent_in_stage_sorted(experiment_dir, stage, agent_num):
    files_for_agent_in_stage = []
    for file in glob(experiment_dir + '/' + stage + '/*'):
        if f"{agent_num}.csv" in file:
            files_for_agent_in_stage.append(file)
    return sort_files_by_date(files_for_agent_in_stage)

def get_date_from_file(file):
    return file.split('/')[-1][1: 11]

def sort_files_by_date(files):
    files.sort(key=lambda file: datetime.strptime(get_date_from_file(file), '%Y-%m-%d'))
    return files

def get_data_from_csv(file, stage_index, file_index): 
    df = pd.read_csv(file)
    df = add_stage_index_and_day_to_df(df, stage_index, file_index)
    df = parse_table_stage_by_odor(df)
    return df


def add_stage_index_and_day_to_df(df, stage_index, file_index):
    df['stage'] = stage_index + 1
    df['day in stage'] = file_index + 1
    return df

def parse_table_stage_by_odor(df): 
    df = set_odor_and_color_column(df)
    df['action'] = df.apply(lambda row: row.chosen_arm, axis=1)
    df['reward'] = df.apply(lambda row: 1 if row.trial_outcome == 1 else 0, axis=1)
    df['reward_type'] = df.apply(lambda row: int(reward_type(row)), axis=1)
    return df[['stage', 'day in stage', 'trial', 'A1o', 'A1c', 'A2o', 'A2c', 'A3o', 'A3c', 'A4o', 'A4c', 'action', 'reward', 'reward_type']]

def set_odor_and_color_column(df):
    df['A1o'] = df.apply(lambda row: 1 if row.correct_p1 == 1 else 0, axis=1)
    df['A1c'] = df.apply(lambda row: 1 if row.irrelevant_stimuli_p1 == 1 else 0, axis=1)
    df['A2o'] = df.apply(lambda row: 1 if row.correct_p1 == 2 else 0, axis=1)
    df['A2c'] = df.apply(lambda row: 1 if row.irrelevant_stimuli_p1 == 2 else 0, axis=1)
    df['A3o'] = df.apply(lambda row: 1 if row.correct_p2 == 3 else 0, axis=1)
    df['A3c'] = df.apply(lambda row: 1 if row.irrelevant_stimuli_p2 == 3 else 0, axis=1)
    df['A4o'] = df.apply(lambda row: 1 if row.correct_p2 == 4 else 0, axis=1)
    df['A4c'] = df.apply(lambda row: 1 if row.irrelevant_stimuli_p2 == 4 else 0, axis=1)
    return df

def reward_type(row):
    if row.action in [1,2]:
        return 1 # food
    elif row.action in [3,4]:
        return 2 # water
    else: 
        return 0


def save_pd_to_csv(df, file_name):
    df.to_csv(file_name, index=False)


expr_data = {1: [1,2], 2: [1], 4: [6,7,8], 5: [1,2], 6: [10, 11]}
for expr in expr_data:
    for rat in expr_data[expr]:
        save_pd_to_csv(get_all_data_from_csv_for_agent(expr,rat), './output_expr{}_rat{}.csv'.format(expr, rat))
 