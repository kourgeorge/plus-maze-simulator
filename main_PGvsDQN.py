__author__ = 'gkour'

import os
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




def main(config, dir_name):
    env = PlusMazeOneHotCues(relevant_cue=config.CueType.ODOR)
    observation_size = env.state_shape()

    repetitions = 2
    agents_DQN_spec = []
    agents_PG_spec = []

    agents_DQN_spec.append([BrainDQN, FullyConnectedNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    agents_PG_spec.append([BrainPG, FullyConnectedNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    agents_DQN_spec.append([BrainDQN, FullyConnectedNetwork2Layers, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    agents_PG_spec.append([BrainPG, FullyConnectedNetwork2Layers, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    agents_DQN_spec.append([BrainDQN, EfficientNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    agents_PG_spec.append([BrainDQN, EfficientNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    agents_DQN_spec.append([BrainDQNFixedDoorAttention, EfficientNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    agents_PG_spec.append([BrainPGFixedDoorAttention, EfficientNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    agents_DQN_spec.append([MotivationDependantBrainDQN, SeparateMotivationAreasNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    agents_PG_spec.append([MotivationDependantBrainPG, SeparateMotivationAreasNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    agents_DQN_spec.append([MotivationDependantBrainDQNLateOutcomeEvaluation, SeparateMotivationAreasNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    agents_PG_spec.append([MotivationDependantBrainPGLateOutcomeEvaluation, SeparateMotivationAreasNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])


    brains_reports = []
    for agent_spec in agents_DQN_spec+agents_PG_spec:
        completed_experiments = 0
        aborted_experiments = 0
        brain_repetition_reports = [None] * repetitions
        while completed_experiments < repetitions:
            agent = MotivatedAgent(agent_spec[0](agent_spec[1](env.stimuli_encoding_size(), 2, env.num_actions())),
                                motivation=config.RewardType.WATER,
                                motivated_reward_value=agent_spec[2], non_motivated_reward_value=agent_spec[3])
            experiment_stats = PlusMazeExperiment(agent, dashboard=False)
            if experiment_stats.metadata['experiment_status'] == EperimentStatus.COMPLETED:
                brain_repetition_reports[completed_experiments] = experiment_stats
                completed_experiments += 1
            else:
                aborted_experiments += 1
        brains_reports.append(brain_repetition_reports)
        print("{} out of {} experiments were aborted".format(aborted_experiments,
                                                            aborted_experiments + completed_experiments))

    plot_days_per_stage(brains_reports, dir_name)
    for brain_report in brains_reports:
        plot_behavior_results(brain_report, dir_name)

    x=1


# class dotdict(dict):
#     __getattr__ = dict.get
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__

from config import gen_get_config

if __name__ == '__main__':
    for config, dirname in gen_get_config():
        os.mkdir('Results/' + dirname)
        main(config, dirname)
        print("Done with {}".format(dirname))
