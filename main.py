from agent import Agent
from environment import PlusMaze
import gym
import numpy as np
import config
from braindqntorch import BrainDQN
from simple_brain import SimpleBrain
from brainddpg import DDPBrain
import utils

reporting_interval = 100

if __name__ == '__main__':
    success_criterion = 0.85
    ############ GYM env %%%%%%%%%%%%%%%%%%%%%%%%
    # env = gym.make('MountainCar-v0')
    # num_actions = env.action_space.n
    # observation_size = env.observation_space.shape[0]
    # agent = Agent(observation_size=observation_size, num_actions=num_actions)

    ############ PlusMaze env %%%%%%%%%%%%%%%%%%%%%%%%
    env = PlusMaze(relevant_cue=config.CueType.ODOR, correct_value=0)
    num_actions = env.num_actions()
    observation_size = env.state_shape()

    # brain = BrainDQN(observation_size=observation_size, num_actions=num_actions, reward_discount=0, learning_rate=1e-4)

    brain = SimpleBrain(num_actions=num_actions, reward_discount=0, learning_rate=config.BASE_LEARNING_RATE)

    # brain = DDPBrain(observation_size,num_actions)
    agent = Agent(brain, motivation=config.RewardType.WATER, motivated_reward_value=1, non_motivated_reward_value=0.3)

    stage = 1
    trial = 0
    act_dist = np.zeros(num_actions)
    loss_acc = 0
    while True:
        trial += 1
        steps, total_reward, act_dist_episode = utils.episode_rollout(env, agent)
        loss = agent.smarten()
        loss_acc += loss

        if trial % reporting_interval == 0:
            report = utils.create_report(agent.get_memory(), reporting_interval)
            print(
                'Trial: {}, Action Dist:{}, Correct:{}, Reward:{}, loss={}'.format(trial,
                                                                                   np.mean(report.action_1hot, axis=0),
                                                                                   np.mean(report.correct),
                                                                                   round(np.mean(report.reward), 2),
                                                                                   round(loss_acc / reporting_interval, 2)),end = '\t')

            water_preference = np.sum(report.arm_type_water)/len(report.arm_type_water)
            water_correct_percent =  np.sum(np.logical_and(report.arm_type_water, report.correct))/np.sum(report.arm_type_water)
            food_correct_percent = np.sum(np.logical_and(np.logical_not(report.arm_type_water), report.correct))/np.sum(np.logical_not(report.arm_type_water))
            print('Water preference:{} Water Correct: {}. Food Correct:{}'.format(water_preference, water_correct_percent, food_correct_percent))
            # print('Trial: {}, Reward:{}'.format(trial, avg_reward))

            if env.stage == 1 and np.mean(report.reward) > success_criterion:
                env.set_odor_options([2, 3])
                env.set_correct_cue_value(3)
                env.stage += 1

                print("Stage {}: Inter-dimensional shift (Odors: {}. Correct {})".format(stage, env._odor_options,
                                                                                         env._correct_cue_value))
            elif env.stage == 2 and np.mean(report.reward) > success_criterion:
                print("Stage 3: Transitioning to food Motivation")
                agent.set_motivation(config.RewardType.FOOD)
                env.stage += 1
            elif env.stage == 3 and np.mean(report.reward) > success_criterion:
                print("Stage 4: Extra-dimensional Shift (Light)")
                env.set_relevant_cue(config.CueType.LIGHT)
                env.set_correct_cue_value(0)
                env.stage += 1
            elif env.stage == 4 and np.mean(report.reward) > success_criterion:
                break

            loss_acc = 0
