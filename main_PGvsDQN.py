__author__ = 'gkour'

from motivatedagent import MotivatedAgent
from environment import PlusMazeOneHotCues

import config
from standardbrainnetwork import StandardBrainNetworkAttention, SeparateNetworkAttention
from braindqn import BrainDQN, BrainDQNFixedDoorAttention, BrainDQNSeparateNetworks
from brainpg import BrainPG, BrainPGFixedDoorAttention, BrainPGSeparateNetworks
from PlusMazeExperiment import PlusMazeExperiment, EperimentStatus
from behavioral_analysis import plot_days_per_stage, plot_behavior_results


if __name__ == '__main__':
    env = PlusMazeOneHotCues(relevant_cue=config.CueType.ODOR)
    num_actions = env.num_actions()
    observation_size = env.state_shape()

    repetitions = 2

    # agents_DQN_spec = [BrainDQNFixedDoorAttention, StandardBrainNetworkAttention, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD]
    # agents_PG_spec = [BrainPGFixedDoorAttention, StandardBrainNetworkAttention, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD]

    agents_DQN_spec = [BrainDQNSeparateNetworks, SeparateNetworkAttention, config.MOTIVATED_REWARD, config.MOTIVATED_REWARD]
    agents_PG_spec = [BrainPGSeparateNetworks, SeparateNetworkAttention, config.MOTIVATED_REWARD, config.MOTIVATED_REWARD]

    brains_reports = []

    for agent_spec in [agents_DQN_spec, agents_PG_spec]:
        completed_experiments = 0
        brain_repetition_reports = [None] * repetitions
        while completed_experiments < repetitions:
            agent = MotivatedAgent(agent_spec[0](agent_spec[1](2, 4)), motivation=config.RewardType.WATER,
                                   motivated_reward_value=agent_spec[2], non_motivated_reward_value=agent_spec[3])
            success, experiment_report_df = PlusMazeExperiment(agent, dashboard=True)
            if success == EperimentStatus.COMPLETED:
                brain_repetition_reports[completed_experiments] = experiment_report_df
                completed_experiments += 1
        brains_reports.append(brain_repetition_reports)

    plot_behavior_results(brains_reports[0])
    plot_behavior_results(brains_reports[1])
    plot_days_per_stage(brains_reports)

    x=1
