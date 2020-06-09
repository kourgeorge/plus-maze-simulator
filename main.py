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
    env = PlusMaze(relevant_cue=config.CueType.ODOR, correct_value=0)
    num_actions = env.num_actions()
    observation_size = env.state_shape()

    #brain = BrainDQN(observation_size=observation_size, num_actions=num_actions, reward_discount=0, learning_rate=1e-4)

    brain = SimpleBrain(num_actions=num_actions, reward_discount=0, learning_rate=config.BASE_LEARNING_RATE)

    agent = Agent(brain, motivation=config.RewardType.WATER, motivated_reward_value=1, non_motivated_reward_value=0)

    stage = 1
    trial = 0
    act_dist = np.zeros(num_actions)
    reward_acc = 0
    loss_acc = 0
    while True:
        trial += 1
        steps, total_reward, act_dist_episode = utils.episode_rollout(env, agent)
        loss = agent.smarten()
        loss_acc += loss
        act_dist += act_dist_episode
        reward_acc += total_reward

        if trial % reporting_interval == 0:
            avg_reward = round(reward_acc / reporting_interval, 2)
            print(
                'Trial: {}, Action Dist:{}, Reward:{}, loss={}'.format(trial, act_dist / reporting_interval, avg_reward,
                                                                       loss_acc / reporting_interval))
            # print('Trial: {}, Reward:{}'.format(trial, avg_reward))

            if stage == 1 and avg_reward > success_criterion:
                env.set_odor_options([2, 3])
                env.set_correct_cue_value(3)
                stage += 1
                print("Stage {}: Inter-dimensional shift (Odors: {}. Correct {})".format(stage, env._odor_options,
                                                                                         env._correct_cue_value))
            elif stage == 2 and avg_reward > success_criterion:
                print("Stage 3: Transitioning to food Motivation")
                agent.set_motivation(config.RewardType.FOOD)
                stage += 1
            elif stage == 3 and avg_reward > success_criterion:
                print("Stage 4: Extra-dimensional Shift (Light)")
                env.set_relevant_cue(config.CueType.LIGHT)
                env.set_correct_cue_value(0)
                stage += 1
            elif stage == 4 and avg_reward > success_criterion:
                break

            act_dist = np.zeros(num_actions)
            reward_acc = 0
            loss_acc = 0
