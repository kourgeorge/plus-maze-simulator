__author__ = 'gkour'

from environment import PlusMaze, PlusMazeOneHotCues
from motivatedagent import MotivatedAgent
import numpy as np
import config
import utils
from Dashboard import Dashboard
from Stats import Stats
import os
import time
from enum import Enum

trials_in_day = 100
max_experiment_length = 50 #days

class EperimentStatus(Enum):
    COMPLETED = 'completed'
    RUNNING = 'running'

def PlusMazeExperiment(agent:MotivatedAgent, dashboard=False):

    env = PlusMazeOneHotCues(relevant_cue=config.CueType.ODOR)
    env.reset()
    stats = Stats(metadata={'brain': str(agent.get_brain()),
                                'network': str(agent.get_brain().get_network()),
                                'brain_params': agent.get_brain().num_trainable_parameters(),
                                'motivated_reward': agent._motivated_reward_value,
                                'non_motivated_reward': agent._non_motivated_reward_value,
                                'experiment_status': EperimentStatus.RUNNING})

    if dashboard:
        dash = Dashboard(agent.get_brain())

    def pre_stage_transition_update():
        if dashboard:
            dash.update(stats.epoch_stats_df, env, agent.get_brain())
            dash.save_fig(dashboard_screenshots_path, env._stage)

    dashboard_screenshots_path = os.path.join('/Users/gkour/repositories/plusmaze/Results', '{}-{}'.format(agent.get_brain(),time.strftime("%Y%m%d-%H%M")))

    trial = 0
    loss_acc = 0
    print('============================ Brain:{} ======================='.format(str(agent.get_brain())))
    print("Stage {}: {} - Water Motivated, odor relevant. (Odors: {}, Correct: {})".format(env._stage, config.stage_names[env._stage], [np.argmax(encoding) for encoding in env.get_odor_cues()],
                                                                                             np.argmax(env.get_correct_cue_value())))

    while env._stage < 5:

        if trial>trials_in_day*max_experiment_length:
            print("Agent failed to learn.")
            return stats

        trial += 1
        utils.episode_rollout(env, agent)

        if trial % trials_in_day == 0:
            loss = agent.smarten()
            report = Stats.create_report_from_memory(agent.get_memory(), agent.get_brain(), trials_in_day)
            stats.update(trial, report)
            pre_stage_transition_update()

            print(
                'Trial: {}, Action Dist:{}, Corr.:{}, Rew.:{}, loss={};'.format(stats.epoch_stats_df['Trial'].to_numpy()[-1],
                                                                                stats.epoch_stats_df['ActionDist'].to_numpy()[-1],
                                                                                stats.epoch_stats_df['Correct'].to_numpy()[-1],
                                                                                stats.epoch_stats_df['Reward'].to_numpy()[-1],
                                                                                round(loss / trials_in_day, 2)))

            print(
                'WPI:{}, WC: {}, FC:{}'.format(stats.epoch_stats_df['WaterPreference'].to_numpy()[-1], stats.epoch_stats_df['WaterCorrect'].to_numpy()[-1],
                                               stats.epoch_stats_df['FoodCorrect'].to_numpy()[-1]))

            current_criterion = np.mean(report.correct)
            reward = np.mean(report.reward)
            if current_criterion > config.SUCCESS_CRITERION_THRESHOLD and reward>0.6:
                set_next_stage(env,agent)

    stats.metadata['experiment_status'] = EperimentStatus.COMPLETED
    return stats


def set_next_stage(env:PlusMaze, agent:MotivatedAgent):
    env._stage += 1
    print('---------------------------------------------------------------------')
    if env._stage==1:
        env.set_random_odor_set()
        print("Stage {}: {} (Odors: {}, Correct:{})".format(env._stage, config.stage_names[env._stage], [np.argmax(encoding) for encoding in env.get_odor_cues()],np.argmax(env.get_correct_cue_value())))
    elif env._stage==2:
        agent.set_motivation(config.RewardType.FOOD)
        print("Stage {}: {}".format(env._stage, config.stage_names[env._stage]))

    elif env._stage==3:
        agent.set_motivation(config.RewardType.WATER)
        env.set_random_odor_set()
        print("Stage {}: {} (Odors: {}. Correct {})".format(env._stage, config.stage_names[env._stage], [np.argmax(encoding) for encoding in env.get_odor_cues()],
                                                                                 np.argmax(env.get_correct_cue_value())))
    elif env._stage==4:
        env.set_relevant_cue(config.CueType.LIGHT)
        print("Stage {}: {} (Lights: {}. Correct {})".format(env._stage, config.stage_names[env._stage], [np.argmax(encoding) for encoding in env.get_light_cues()],
                                                                                 np.argmax(env.get_correct_cue_value())))