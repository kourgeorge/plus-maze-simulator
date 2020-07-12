from agent import Agent
from environment import PlusMaze

import numpy as np
import config
from table_dqn_brain import TableDQNBrain
from brainpg import BrainPG
from brainac import BrainAC
import utils
from Dashboard import Dashboard
from Animation import Animation
from Stats import Stats

reporting_interval = 100
if __name__ == '__main__':

    ############ GYM env %%%%%%%%%%%%%%%%%%%%%%%%
    # import gym
    # env = gym.make('MountainCar-v0')
    # num_actions = env.action_space.n
    # observation_size = env.observation_space.shape[0]
    # agent = Agent(observation_size=observation_size, num_actions=num_actions)

    ############ PlusMaze env %%%%%%%%%%%%%%%%%%%%%%%%
    env = PlusMaze(relevant_cue=config.CueType.ODOR)
    num_actions = env.num_actions()
    observation_size = env.state_shape()


    #env.set_odor_options([[-2], [2]])
    #env.set_correct_cue_value([2])

    # brain = TableDQNBrain(num_actions=num_actions, reward_discount=0, learning_rate=config.BASE_LEARNING_RATE)

    brain = BrainPG(observation_size, num_actions, reward_discount=0, learning_rate=config.LEARNING_RATE)
    agent = Agent(brain, motivation=config.RewardType.WATER, motivated_reward_value=config.MOTIVATED_REWARD, non_motivated_reward_value=config.NON_MOTIVATED_REWARD)

    stats = Stats()
    dash = Dashboard(brain)
    anim = Animation(dash.get_fig(), "result")

    trial = 0
    act_dist = np.zeros(num_actions)
    loss_acc = 0
    print("Stage 1: Baseline - Water Motivated, odor relevant. (Odors: {}, Correct: {})".format(env._odor_options,
                                                                                                env._correct_cue_value))

    while True:
        trial += 1
        steps, total_reward, act_dist_episode = utils.episode_rollout(env, agent)
        loss = agent.smarten()
        loss_acc += loss

        if trial % reporting_interval == 0:

            report = utils.create_report(agent.get_memory(), reporting_interval)
            epoch_stats_df = stats.update(trial, report)
            print(
                'Trial: {}, Action Dist:{}, Corr.:{}, Rew.:{}, loss={};'.format(epoch_stats_df['Trial'].to_numpy()[-1],
                                                                                     epoch_stats_df['ActionDist'].to_numpy()[-1],
                                                                                     epoch_stats_df['Correct'].to_numpy()[-1],
                                                                                     epoch_stats_df['Reward'].to_numpy()[-1],
                                                                                     round(
                        loss_acc / reporting_interval,
                        2)), end='\t')

            print(
                'WPI:{}, WC: {}, FC:{}'.format(epoch_stats_df['WaterPreference'].to_numpy()[-1], epoch_stats_df['WaterCorrect'].to_numpy()[-1],
                                               epoch_stats_df['FoodCorrect'].to_numpy()[-1]))

            # visualize
            dash.update(epoch_stats_df, env, brain)
            anim.add_frame()

            current_criterion = np.mean(report.reward)
            if env.stage == 1 and current_criterion > config.SUCCESS_CRITERION_THRESHOLD:
                #env.set_odor_options([[-2],[2]])
                env.set_odor_options([[0.6, 0.5], [0.1, 0.7]])
                #env.set_correct_cue_value([2])
                env.set_correct_cue_value([0.1, 0.7])
                env.stage += 1
                print("Stage {}: Inter-dimensional shift (Odors: {}. Correct {})".format(env.stage, env._odor_options,
                                                                                         env._correct_cue_value))

                brain.policy.controller.reset_parameters()
                brain.policy.affine.reset_parameters()

            elif env.stage == 2 and current_criterion > config.SUCCESS_CRITERION_THRESHOLD:
                print("Stage 3: Transitioning to food Motivation")
                agent.set_motivation(config.RewardType.FOOD)
                env.stage += 1
           #     brain.policy.l2.reset_parameters()
            elif env.stage == 3 and current_criterion > config.SUCCESS_CRITERION_THRESHOLD:
                print("Stage 4: Extra-dimensional Shift (Light)")
                agent.set_motivation(config.RewardType.WATER)
                env.set_relevant_cue(config.CueType.LIGHT)
                #env.set_odor_options([[3], [-3]])
                env.set_correct_cue_value([0.4, 0.2])
                env.stage += 1
            elif env.stage == 4 and current_criterion > config.SUCCESS_CRITERION_THRESHOLD:
                break

            loss_acc = 0
