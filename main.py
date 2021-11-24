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
reporting_interval = 100


def PlusMazeExperiment(agent, dashboard=False):

    env = PlusMaze(relevant_cue=config.CueType.ODOR)
    stats = Stats()

    def pre_stage_transition_update():
        if dashboard:
            dash = Dashboard(brain)
            dash.update(epoch_stats_df, env, brain)
            dash.save_fig(results_path, env.stage)

    results_path = os.path.join('/Users/gkour/repositories/plusmaze/Results', '{}-{}'.format(brain,time.strftime("%Y%m%d-%H%M")))

    trial = 0
    loss_acc = 0
    print("Stage 1: Baseline - Water Motivated, odor relevant. (Odors: {}, Correct: {})".format(env._odor_options,
                                                                                                env._correct_cue_value))
    while True:
        trial += 1
        utils.episode_rollout(env, agent)
        loss = agent.smarten()
        loss_acc += loss

        if trial % reporting_interval == 0:
            report = utils.create_report_from_memory(agent.get_memory(), reporting_interval)
            epoch_stats_df = stats.update(trial, report)
            print(
                'Trial: {}, Action Dist:{}, Corr.:{}, Rew.:{}, loss={};'.format(epoch_stats_df['Trial'].to_numpy()[-1],
                                                                                     epoch_stats_df['ActionDist'].to_numpy()[-1],
                                                                                     epoch_stats_df['Correct'].to_numpy()[-1],
                                                                                     epoch_stats_df['Reward'].to_numpy()[-1],
                                                                                     round(loss_acc / reporting_interval,2)))

            print(
                'WPI:{}, WC: {}, FC:{}'.format(epoch_stats_df['WaterPreference'].to_numpy()[-1], epoch_stats_df['WaterCorrect'].to_numpy()[-1],
                                               epoch_stats_df['FoodCorrect'].to_numpy()[-1]))



            current_criterion = np.mean(report.correct)
            if env.stage == 1 and current_criterion > config.SUCCESS_CRITERION_THRESHOLD:


                #env.set_odor_options([[-2],[2]])
                env.set_odor_options([[0.5, 0.2], [0.9, 0.7]])
                #env.set_correct_cue_value([2])
                env.set_correct_cue_value([0.9, 0.7])
                env.stage += 1
                print("Stage {}: Inter-dimensional shift (Odors: {}. Correct {})".format(env.stage, env._odor_options,
                                                                                         env._correct_cue_value))

                #brain.policy.controller.reset_parameters()

            elif env.stage == 2 and current_criterion > config.SUCCESS_CRITERION_THRESHOLD:
                pre_stage_transition_update()
                print("Stage 3: Transitioning to food Motivation")
                agent.set_motivation(config.RewardType.FOOD)
                env.stage += 1
           #     brain.policy.l2.reset_parameters()
            elif env.stage == 3 and current_criterion > config.SUCCESS_CRITERION_THRESHOLD:
                pre_stage_transition_update()
                print("Stage 4: Back to Water to Motivation")
                agent.set_motivation(config.RewardType.WATER)
                env.stage += 1
                #env.set_odor_options([[0.5, 0.1], [0.8, 0.3]])
                #env.set_odor_options([[3], [-3]])
                #env.set_correct_cue_value([0.5, 0.1])

                #brain.policy.controller.reset_parameters()

            elif env.stage == 4 and current_criterion > config.SUCCESS_CRITERION_THRESHOLD:
                pre_stage_transition_update()
                print("Stage 5: Extra-dimensional Shift (Light)")
                agent.set_motivation(config.RewardType.WATER)
                env.set_relevant_cue(config.CueType.LIGHT)
                env.set_odor_options([[0.02, 0.4], [0.8, 0.8]])
                #env.set_odor_options([[3], [-3]])
                env.set_correct_cue_value([0.4, 0.2])
                env.stage += 1

                #brain.policy.controller.reset_parameters()
            elif env.stage == 5 and current_criterion > config.SUCCESS_CRITERION_THRESHOLD:
                pre_stage_transition_update()
                break

            loss_acc = 0

    return epoch_stats_df


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
    print(np.mean(days_per_stage))
    print(np.std(days_per_stage))
