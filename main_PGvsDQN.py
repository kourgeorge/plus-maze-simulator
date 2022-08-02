__author__ = 'gkour'

import os
import random
import time

import pandas as pd
from config import MOTIVATED_REWARD
from fixeddoorattentionbrain import BrainDQNFixedDoorAttention, BrainPGFixedDoorAttention
from lateoutcomeevaluationbrain import MotivationDependantBrainDQNLateOutcomeEvaluation, MotivationDependantBrainPGLateOutcomeEvaluation
from motivatedagent import MotivatedAgent
from environment import PlusMazeOneHotCues

# import config
from motivationdependantbrain import MotivationDependantBrainDQN, MotivationDependantBrainPG
from standardbrainnetwork import FullyConnectedNetwork, EfficientNetwork, SeparateMotivationAreasNetwork, \
    FullyConnectedNetwork2Layers
from braindqn import BrainDQN
from brainpg import BrainPG
from PlusMazeExperiment import PlusMazeExperiment, EperimentStatus
from behavioral_analysis import plot_days_per_stage, plot_behavior_results



def main(config, dir_name, rat_data_file = None, rat_id = None, df = None):
    env = PlusMazeOneHotCues(relevant_cue=config.CueType.ODOR)
    observation_size = env.state_shape()
    num_trials = len((pd.read_csv(rat_data_file)).index)

    repetitions = 3
    agents_DQN_spec = []
    agents_PG_spec = []

    agents_DQN_spec.append([BrainDQN, FullyConnectedNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    # agents_PG_spec.append([BrainPG, FullyConnectedNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    agents_DQN_spec.append([BrainDQN, FullyConnectedNetwork2Layers, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    # agents_PG_spec.append([BrainPG, FullyConnectedNetwork2Layers, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    agents_DQN_spec.append([BrainDQN, EfficientNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    # agents_PG_spec.append([BrainDQN, EfficientNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    agents_DQN_spec.append([BrainDQNFixedDoorAttention, EfficientNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    # agents_PG_spec.append([BrainPGFixedDoorAttention, EfficientNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    agents_DQN_spec.append([MotivationDependantBrainDQN, SeparateMotivationAreasNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    # agents_PG_spec.append([MotivationDependantBrainPG, SeparateMotivationAreasNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    agents_DQN_spec.append([MotivationDependantBrainDQNLateOutcomeEvaluation, SeparateMotivationAreasNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    # agents_PG_spec.append([MotivationDependantBrainPGLateOutcomeEvaluation, SeparateMotivationAreasNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    
    brains_reports = []
    for agent_spec in agents_DQN_spec:
        completed_experiments = 0
        brain_repetition_reports = [None] * repetitions
        for completed_experiments in range(repetitions):
            agent = MotivatedAgent(agent_spec[0](agent_spec[1](env.stimuli_encoding_size(), 2, env.num_actions()), learning_rate = config.LEARNING_RATE ),
                                motivation=config.RewardType.WATER,
                                motivated_reward_value=agent_spec[2], non_motivated_reward_value=agent_spec[3])
            experiment_stats, likelihood = PlusMazeExperiment(agent, dashboard=False, rat_data_file=rat_data_file)
            
            dict = {'rat': rat_id, 'brain': agent_spec[0].__name__, 'network': agent_spec[1].__name__, 'memory_size': config.MEMORY_SIZE,
                     'forgetting': config.FORGETTING, 'motivated_reward': config.MOTIVATED_REWARD, 'non_motivated_reward': config.NON_MOTIVATED_REWARD, 'likelihood': likelihood, 'trials': num_trials,
                     'param_number': agent.get_brain().num_trainable_parameters(), 'repetition': completed_experiments}
            df = df.append(dict, ignore_index=True)
            if experiment_stats.metadata['experiment_status'] == EperimentStatus.COMPLETED:
                brain_repetition_reports[completed_experiments] = experiment_stats
        brains_reports.append(brain_repetition_reports)

    
    plot_days_per_stage(brains_reports, file_path = 'Results/{}/days_in_stage_{}'.format(dir_name, time.strftime("%Y%m%d-%H%M")))
    for brain_report in brains_reports:
        plot_behavior_results(brain_report, dir_name)

    x=1
    return df


from config import gen_get_config

def get_time_YYYY_MM_DD_HH_MM():
    import datetime
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

def init_df():
    df = pd.DataFrame(columns=['rat', 'brain', 'network', 'memory_size', 'forgetting', 'motivated_reward', 'non_motivated_reward', 'likelihood', 'trials', 'param_number'])
    return df

expr_data = {1: [1,2], 2: [1], 4: [6,7,8], 5: [1,2], 6: [10, 11]}

if __name__ == '__main__':
    df = init_df()
    for config, dirname in gen_get_config():
        os.mkdir('Results/' + dirname)
        for expr in expr_data:
            for rat in expr_data[expr]:
                df = main(config, dirname, './output_expr{}_rat{}.csv'.format(expr, rat), '{}_{}'.format(expr, rat), df)
                df.to_csv('output_until_expr{}_rat{}.csv'.format(expr, rat))
        print("Done with {}".format(dirname))
    df.to_csv('outputForAll.csv')
