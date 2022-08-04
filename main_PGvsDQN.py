__author__ = 'gkour'

import os
import random
import time

import pandas as pd
from config import MOTIVATED_REWARD
from lateoutcomeevaluationbrain import LateOutcomeEvaluationBrain
from motivatedagent import MotivatedAgent
from environment import PlusMazeOneHotCues

# import config
from standardbrainnetwork import FullyConnectedNetwork, EfficientNetwork, SeparateMotivationAreasNetwork, \
    FullyConnectedNetwork2Layers
from learner import DQN, PG
from fixeddoorattentionbrain import FixedDoorAttentionBrain
from motivationdependantbrain import MotivationDependantBrain
from PlusMazeExperiment import PlusMazeExperiment, EperimentStatus
from behavioral_analysis import plot_days_per_stage, plot_behavior_results
from consolidationbrain import ConsolidationBrain


def main(config, dir_name, rat_data_file = None, rat_id = None, df = None):
    env = PlusMazeOneHotCues(relevant_cue=config.CueType.ODOR)
    observation_size = env.state_shape()
    num_trials = len((pd.read_csv(rat_data_file)).index)

    repetitions = 3
    agents_DQN_spec = []
    agents_PG_spec = []

    agents_DQN_spec.append([ConsolidationBrain, DQN, FullyConnectedNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    # agents_PG_spec.append([ConsolidationBrain, PG, FullyConnectedNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    agents_DQN_spec.append([ConsolidationBrain, DQN, FullyConnectedNetwork2Layers, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    # agents_PG_spec.append([ConsolidationBrain, PG, FullyConnectedNetwork2Layers, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    agents_DQN_spec.append([ConsolidationBrain, DQN, EfficientNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    # agents_PG_spec.append([ConsolidationBrain, PG, EfficientNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    agents_DQN_spec.append([FixedDoorAttentionBrain, DQN, EfficientNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    # agents_PG_spec.append([FixedDoorAttentionBrain, PG, EfficientNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    agents_DQN_spec.append([MotivationDependantBrain, DQN, SeparateMotivationAreasNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    # agents_PG_spec.append([MotivationDependantBrain, PG, SeparateMotivationAreasNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    agents_DQN_spec.append([LateOutcomeEvaluationBrain, DQN, SeparateMotivationAreasNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    # agents_PG_spec.append([LateOutcomeEvaluationBrain, PG, SeparateMotivationAreasNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    
    brains_reports = []
    for agent_spec in agents_DQN_spec:
        completed_experiments = 0
        brain_repetition_reports = [None] * repetitions
        for completed_experiments in range(repetitions):
            (brain, learner, network, motivated_reward_value, non_motivated_reward_value) = agent_spec 
            agent = MotivatedAgent(brain(learner(network(env.stimuli_encoding_size(), 2, env.num_actions()), learning_rate=config.LEARNING_RATE)),
                                   motivation=config.RewardType.WATER,
                                   motivated_reward_value=motivated_reward_value, non_motivated_reward_value=non_motivated_reward_value)
            experiment_stats, likelihood = PlusMazeExperiment(agent, dashboard=False, rat_data_file=rat_data_file)
            
            dict = {'rat': rat_id, 'brain': agent_spec[0].__name__, 'network': agent_spec[1].__name__,
                     'forgetting': config.FORGETTING, 'motivated_reward': config.MOTIVATED_REWARD, 'non_motivated_reward': config.NON_MOTIVATED_REWARD, 'memory_size': config.MEMORY_SIZE, 'learning_rate': config.LEARNING_RATE, 'likelihood': likelihood, 'trials': int(num_trials),
                     'param_number': agent.get_brain().num_trainable_parameters(), 'repetition': int(completed_experiments)}
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
    df = pd.DataFrame(columns=['rat', 'brain', 'network', 'forgetting', 'motivated_reward', 'non_motivated_reward', 'memory_size', 'learning_rate', 'likelihood', 'trials', 'param_number'])
    return df

# expr_data = {1: [1,2], 2: [1], 4: [6,7,8], 5: [1,2], 6: [10, 11]}
expr_data = {1: [1,2], 2: [1], 4: [6,7]}


if __name__ == '__main__':
    df = init_df()
    for config_index,(config, dirname) in enumerate(gen_get_config()):
        if not os.path.exists('Results/' + dirname):
            os.makedirs('Results/' + dirname)
        for expr in expr_data:
            for rat in expr_data[expr]:
                df = main(config, dirname, './output_expr{}_rat{}.csv'.format(expr, rat), '{}_{}'.format(expr, rat), df)
                df.to_csv('Rats-Results/output_until_expr{}_rat{}_config_{}.csv'.format(expr, rat, config_index))
        print("Done with {}".format(dirname))
    df.to_csv('outputForAll.csv')
