from agent import Agent
from environment import PlusMaze

import numpy as np
import config
from table_dqn_brain import TableDQNBrain
from brainpg import BrainPG
from brainac import BrainAC
from braindqn import BrainDQN
import utils
from Dashboard import Dashboard
from Stats import Stats
import os
import time
from collections import Counter
from PlusMazeExperiment import PlusMazeExperiment

reporting_interval = 100


if __name__ == '__main__':
    env = PlusMaze(relevant_cue=config.CueType.ODOR)
    num_actions = env.num_actions()
    observation_size = env.state_shape()

    repetitions = 5

    days_per_stage = []
    for rep in range(repetitions):
        # brain = TableDQNBrain(num_actions=num_actions, reward_discount=0, learning_rate=config.LEARNING_RATE)
        #brain = BrainPG(observation_size + 1, num_actions, reward_discount=0, learning_rate=config.LEARNING_RATE)
        brain = BrainDQN (observation_size+1, num_actions, reward_discount=0, learning_rate=config.LEARNING_RATE)
        agent = Agent(brain, motivation=config.RewardType.WATER, motivated_reward_value=config.MOTIVATED_REWARD,
                      non_motivated_reward_value=config.NON_MOTIVATED_REWARD)

        experiment_report_df = PlusMazeExperiment(agent, dashboard=False)
        c = Counter(list(experiment_report_df['Stage']))
        days_per_stage.append([c[i] for i in range(1,5)])

    days_per_stage = np.stack(days_per_stage)
    print(np.mean(days_per_stage, axis=0))
    print(np.std(days_per_stage, axis=0))
