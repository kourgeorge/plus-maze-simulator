from agent import Agent
from environment import PlusMaze
import gym
import numpy as np
import config
from braindqntorch import BrainDQN
from simple_brain import SimpleBrain
import utils

reporting_interval = 100

if __name__ == '__main__':
    success_criterion = 0.9
    ############ GYM env %%%%%%%%%%%%%%%%%%%%%%%%
    # env = gym.make('MountainCar-v0')
    # num_actions = env.action_space.n
    # observation_size = env.observation_space.shape[0]
    # agent = Agent(observation_size=observation_size, num_actions=num_actions)

    ############ PlusMaze env %%%%%%%%%%%%%%%%%%%%%%%%
    env = PlusMaze(relevant_cue=config.CueType.ODOR, correct_value=2)
    num_actions = env.num_actions()
    observation_size = env.state_shape()

    # brain = BrainDQN(observation_size=observation_size,
    #                        num_actions=num_actions,
    #                        reward_discount=0,
    #                        learning_rate=config.BASE_LEARNING_RATE)

    brain = SimpleBrain(num_actions=num_actions,
                        reward_discount=0,
                        learning_rate=config.BASE_LEARNING_RATE)

    agent = Agent(brain, motivation=config.RewardType.WATER, motivated_reward_value=1, non_motivated_reward_value=0)

    stage = 1
    trial = 0
    act_dist = np.zeros(num_actions)
    reward_acc = 0
    while True:
        trial += 1
        steps, total_reward, act_dist_episode = utils.episode_rollout(env, agent)
        loss = agent.smarten()

        act_dist += act_dist_episode
        reward_acc += total_reward

        if trial % reporting_interval == 0:
            print('Trial: {}, Action Dist:{}, Reward:{}'.format(trial, act_dist / reporting_interval, reward_acc / reporting_interval))

            if stage == 1 and (reward_acc / reporting_interval) > success_criterion:
                stage += 1
                print("Stage {}: Inter-dimensional shift".format(stage))
                env = PlusMaze(relevant_cue=config.CueType.ODOR, correct_value=2)
            elif stage == 2 and (reward_acc / reporting_interval) > success_criterion:
                stage += 1
                print("Stage 3: Transitioning to food Motivation")
                agent.set_motivation(config.RewardType.FOOD)
            elif stage == 3 and (reward_acc / reporting_interval) > success_criterion:
                break

            act_dist = np.zeros(num_actions)
            reward_acc = 0
