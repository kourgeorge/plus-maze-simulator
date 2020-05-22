from agent import Agent
from environment import PlusMaze
import gym
import numpy as np
import config

reporting_interval = 500


def episode_rollout(env, agent):
    total_reward = 0

    # num_actions = env.action_space.n
    num_actions = env.num_actions()

    act_dist = np.zeros(num_actions)

    state = env.reset()
    terminated = False
    steps = 0
    while not terminated:
        steps += 1
        # env.render()
        action = agent.decide(state)
        dec_1hot = np.zeros(num_actions)
        dec_1hot[action] = 1
        act_dist += dec_1hot
        new_state, reward, terminated, info = env.step(action)
        reward = agent.evaluate_reward(reward)
        total_reward += reward
        agent.add_experience(state, dec_1hot, reward, new_state, terminated)
        state = new_state

    return steps, total_reward, act_dist


if __name__ == '__main__':

    ############ GYM env %%%%%%%%%%%%%%%%%%%%%%%%
    # env = gym.make('MountainCar-v0')
    # num_actions = env.action_space.n
    # observation_size = env.observation_space.shape[0]
    # agent = Agent(observation_size=observation_size, num_actions=num_actions)

    ############ PlusMaze env %%%%%%%%%%%%%%%%%%%%%%%%
    env = PlusMaze(relevant_cue=config.CueType.ODOR, correct_value=1)
    num_actions = env.num_actions()
    observation_size = env.state_shape()
    agent = Agent(motivation=config.RewardType.WATER, motivated_reward_value=1, non_motivated_reward_value=1)

    act_dist = np.zeros(num_actions)
    reward_acc = 0
    for trial in range(1, 60000):
        steps, total_reward, act_dist_episode = episode_rollout(env, agent)
        loss = agent.smarten()

        act_dist += act_dist_episode
        reward_acc += total_reward

        if trial % reporting_interval == 0:
            print('Action Dist:{}, Reward:{}'.format(act_dist / reporting_interval, reward_acc / reporting_interval))
            act_dist = np.zeros(num_actions)
            reward_acc = 0

        # if trial == 35000:
        #     print("Stage 2: change the rewarded value")
        #     env = PlusMaze(relevant_cue=config.CueType.ODOR, correct_value=-1)
        # if trial == 45000:
        #     print("Stage 3: Transitioning to food Motivation")
        #     agent.set_motivation(config.RewardType.FOOD)
