from agent import Agent
from environment import PlusMaze

import config
from table_dqn_brain import TableDQNBrain
from brainpg import BrainPG
from brainac import BrainAC
from braindqn import BrainDQN
from PlusMazeExperiment import PlusMazeExperiment
import matplotlib.pyplot as plt
from behavioral_analysis import plot_days_per_stage, days_to_consider_in_each_stage


if __name__ == '__main__':
    env = PlusMaze(relevant_cue=config.CueType.ODOR)
    num_actions = env.num_actions()
    observation_size = env.state_shape()

    repetitions = 3
    reports_dqn = [None]*repetitions
    reports_pg = [None]*repetitions

    for rep in range(repetitions):
        # brain = TableDQNBrain(num_actions=num_actions, reward_discount=0, learning_rate=config.LEARNING_RATE)
        brain = BrainPG(observation_size + 1, num_actions, reward_discount=0, learning_rate=config.LEARNING_RATE)
        agent = Agent(brain, motivation=config.RewardType.WATER, motivated_reward_value=config.MOTIVATED_REWARD,
                      non_motivated_reward_value=config.NON_MOTIVATED_REWARD)

        experiment_report_df_pg = PlusMazeExperiment(agent, dashboard=False)
        reports_pg[rep] = experiment_report_df_pg
        brain = BrainDQN (observation_size+1, num_actions, reward_discount=0, learning_rate=config.LEARNING_RATE)
        agent = Agent(brain, motivation=config.RewardType.WATER, motivated_reward_value=config.MOTIVATED_REWARD,
                      non_motivated_reward_value=config.NON_MOTIVATED_REWARD)
        experiment_report_df_dqn = PlusMazeExperiment(agent, dashboard=False)
        reports_dqn[rep] = experiment_report_df_dqn

    days_to_consider_in_each_stage(reports_pg)
    plot_days_per_stage(reports_pg, reports_dqn)

