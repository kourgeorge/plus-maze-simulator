from agent import Agent
from environment import PlusMaze

import config
from table_dqn_brain import TableDQNBrain
from brainpg import BrainPG
from brainac import BrainAC
from braindqn import BrainDQN
from PlusMazeExperiment import PlusMazeExperiment
import matplotlib.pyplot as plt
from behavioral_analysis import plot_days_per_stage, plot_behavior_results


if __name__ == '__main__':
    env = PlusMaze(relevant_cue=config.CueType.ODOR)
    num_actions = env.num_actions()
    observation_size = env.state_shape()

    repetitions = 25

    brain_types = [BrainDQN,BrainPG]
    brains_reports = []
    for brain_type in brain_types:
        brain_repetition_reports = [None] * repetitions
        for rep in range(repetitions):
            brain = brain_type(observation_size+1, num_actions, reward_discount=0, learning_rate=config.LEARNING_RATE)
            agent = Agent(brain, motivation=config.RewardType.WATER, motivated_reward_value=config.MOTIVATED_REWARD,
                          non_motivated_reward_value=config.NON_MOTIVATED_REWARD)
            experiment_report_df_dqn = PlusMazeExperiment(agent, dashboard=False)
            brain_repetition_reports[rep] = experiment_report_df_dqn
        brains_reports.append(brain_repetition_reports)

    plot_behavior_results(brains_reports[0])
    plot_behavior_results(brains_reports[1])
    plot_days_per_stage(brains_reports)

    x=1
