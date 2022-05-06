__author__ = 'gkour'

from fixeddoorattentionbrain import BrainDQNFixedDoorAttention, BrainPGFixedDoorAttention
from lateoutcomeevaluationbrain import MotivationDependantBrainDQNLateOutcomeEvaluation, MotivationDependantBrainPGLateOutcomeEvaluation
from motivatedagent import MotivatedAgent
from environment import PlusMazeOneHotCues

import config
from motivationdependantbrain import MotivationDependantBrainDQN, MotivationDependantBrainPG
from standardbrainnetwork import FullyConnectedNetwork, DoorAttentionAttention, SeparateMotivationAreasNetwork, \
    FullyConnectedNetwork2Layers
from braindqn import BrainDQN
from brainpg import BrainPG
from PlusMazeExperiment import PlusMazeExperiment, EperimentStatus
from behavioral_analysis import plot_days_per_stage, plot_behavior_results


if __name__ == '__main__':
    env = PlusMazeOneHotCues(relevant_cue=config.CueType.ODOR)
    observation_size = env.state_shape()

    repetitions = 10
    agents_DQN_spec=[]
    agents_PG_spec= []

    agents_DQN_spec.append([BrainDQN, FullyConnectedNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    agents_PG_spec.append([BrainPG, FullyConnectedNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    agents_DQN_spec.append([MotivationDependantBrainDQN, SeparateMotivationAreasNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    agents_PG_spec.append([MotivationDependantBrainPG, SeparateMotivationAreasNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    agents_DQN_spec.append([BrainDQNFixedDoorAttention, DoorAttentionAttention, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    agents_PG_spec.append([BrainPGFixedDoorAttention, DoorAttentionAttention, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    #
    agents_DQN_spec.append([MotivationDependantBrainDQNLateOutcomeEvaluation, SeparateMotivationAreasNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])
    agents_PG_spec.append([MotivationDependantBrainPGLateOutcomeEvaluation, SeparateMotivationAreasNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD])

    agent_specs = [x for y in zip(agents_DQN_spec, agents_PG_spec) for x in y]

    brains_reports = []
    for agent_spec in agent_specs:
        completed_experiments = 0
        aborted_experiments = 0
        brain_repetition_reports = [None] * repetitions
        while completed_experiments < repetitions:
            agent = MotivatedAgent(agent_spec[0](agent_spec[1](env.stimuli_encoding_size, 2, env.num_actions())),
                                   motivation=config.RewardType.WATER,
                                   motivated_reward_value=agent_spec[2], non_motivated_reward_value=agent_spec[3])
            experiment_stats = PlusMazeExperiment(agent, dashboard=True)
            if experiment_stats.metadata['experiment_status'] == EperimentStatus.COMPLETED:
                brain_repetition_reports[completed_experiments] = experiment_stats
                completed_experiments += 1
            else:
                aborted_experiments += 1
        brains_reports.append(brain_repetition_reports)
        print("{} out of {} experiments were aborted".format(aborted_experiments,
                                                             aborted_experiments + completed_experiments))

    plot_days_per_stage(brains_reports)
    for brain_report in brains_reports:
        plot_behavior_results(brain_report)

    x=1
