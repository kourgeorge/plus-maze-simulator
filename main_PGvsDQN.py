__author__ = 'gkour'

import os.path

from lateoutcomeevaluationbrain import LateOutcomeEvaluationBrain
from motivatedagent import MotivatedAgent
from environment import PlusMazeOneHotCues, CueType

import os
import config
from standardbrainnetwork import FullyConnectedNetwork, EfficientNetwork, SeparateMotivationAreasNetwork, \
    FullyConnectedNetwork2Layers, TabularQ, UniformAttentionTabular
from learner import TD, DQN, PG, TDUniformAttention
from fixeddoorattentionbrain import FixedDoorAttentionBrain
from motivationdependantbrain import MotivationDependantBrain
from PlusMazeExperiment import PlusMazeExperiment, ExperimentStatus
from behavioral_analysis import plot_days_per_stage, plot_behavior_results
from consolidationbrain import ConsolidationBrain
from rewardtype import RewardType
from tdbrain import TDBrain

if __name__ == '__main__':
    env = PlusMazeOneHotCues(relevant_cue=CueType.ODOR)
    observation_size = env.state_shape()

    repetitions = 2
    agents_DQN_spec = []
    agents_PG_spec = []

    agents_DQN_spec.append([TDBrain, TD, TabularQ, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    agents_DQN_spec.append([TDBrain, TDUniformAttention, UniformAttentionTabular, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    agents_DQN_spec.append([ConsolidationBrain, DQN, FullyConnectedNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    agents_PG_spec.append([ConsolidationBrain, PG, FullyConnectedNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    agents_DQN_spec.append([ConsolidationBrain, DQN, FullyConnectedNetwork2Layers, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    agents_PG_spec.append([ConsolidationBrain, PG, FullyConnectedNetwork2Layers, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    agents_DQN_spec.append([ConsolidationBrain, DQN, EfficientNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    agents_PG_spec.append([ConsolidationBrain, PG, EfficientNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    agents_DQN_spec.append([FixedDoorAttentionBrain, DQN, EfficientNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    agents_PG_spec.append([FixedDoorAttentionBrain, PG, EfficientNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    agents_DQN_spec.append([MotivationDependantBrain, DQN, SeparateMotivationAreasNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    agents_PG_spec.append([MotivationDependantBrain, PG, SeparateMotivationAreasNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    agents_DQN_spec.append([LateOutcomeEvaluationBrain, DQN, SeparateMotivationAreasNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    agents_PG_spec.append([LateOutcomeEvaluationBrain, PG, SeparateMotivationAreasNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    brains_reports = []
    for agent_spec in agents_DQN_spec+agents_PG_spec:
        completed_experiments = 0
        aborted_experiments = 0
        brain_repetition_reports = [None] * repetitions
        while completed_experiments < repetitions:
            (brain, learner, network, motivated_reward_value, non_motivated_reward_value) = agent_spec
            agent = MotivatedAgent(brain(learner(network(env.stimuli_encoding_size(), 2, env.num_actions()), learning_rate=config.LEARNING_RATE)),
                                   motivation=RewardType.WATER,
                                   motivated_reward_value=motivated_reward_value, non_motivated_reward_value=non_motivated_reward_value)
            experiment_stats = PlusMazeExperiment(agent, dashboard=False)
            if experiment_stats.metadata['experiment_status'] == ExperimentStatus.COMPLETED:
                brain_repetition_reports[completed_experiments] = experiment_stats
                completed_experiments += 1
            else:
                aborted_experiments += 1
        brains_reports.append(brain_repetition_reports)
        print("{} out of {} experiments were aborted".format(aborted_experiments,
                                                             aborted_experiments + completed_experiments))

    plot_days_per_stage(brains_reports, file_path=os.path.join('Results', 'days_per_stage.png'))

    for brain_report in brains_reports:
        plot_behavior_results(brain_report)

    x=1
