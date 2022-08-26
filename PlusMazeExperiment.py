__author__ = 'gkour'

from environment import PlusMaze, PlusMazeOneHotCues, CueType
from motivatedagent import MotivatedAgent
from rewardtype import RewardType
import numpy as np
import config
import utils
from Dashboard import Dashboard
from Stats import Stats
import os
import time
from enum import Enum

stage_names = ['Baseline', 'IDshift', 'Mshift(Food)', 'MShift(Water)+IDshift', 'EDShift(Light)', 'EDshift(Spatial)']

trials_in_day = 100
max_experiment_length = len(stage_names)*10 #days

class ExperimentStatus(Enum):
    COMPLETED = 'completed'
    RUNNING = 'running'

def PlusMazeExperiment(agent:MotivatedAgent, dashboard=False):

    env = PlusMazeOneHotCues(relevant_cue=CueType.ODOR)
    env.reset()
    stats = Stats(metadata={'brain': str(agent.get_brain()),
                                'network': str(agent.get_brain().get_network()),
                                'brain_params': agent.get_brain().num_trainable_parameters(),
                                'motivated_reward': agent._motivated_reward_value,
                                'non_motivated_reward': agent._non_motivated_reward_value,
                                'experiment_status': ExperimentStatus.RUNNING})

    if dashboard:
        dash = Dashboard(agent.get_brain())

    def pre_stage_transition_update():
        if dashboard:
            dash.update(stats.epoch_stats_df, env, agent.get_brain())
            dash.save_fig(dashboard_screenshots_path, env._stage)

    dashboard_screenshots_path = os.path.join('/Users/gkour/repositories/plusmaze/Results', '{}-{}'.format(agent.get_brain(),time.strftime("%Y%m%d-%H%M")))

    trial = 0
    print('============================ Brain:{}, Network:{} ======================='.format(str(agent.get_brain()),str(agent.get_brain().get_network())))
    print("Stage {}: {} - Water Motivated, odor relevant. (Odors: {}, Correct: {})".format(env._stage, stage_names[env._stage], [np.argmax(encoding) for encoding in env.get_odor_cues()],
                                                                                             np.argmax(env.get_correct_cue_value())))

    while env._stage < len(stage_names):

        if trial>trials_in_day*max_experiment_length:
            print("Agent failed to learn.")
            return stats

        trial += 1
        utils.episode_rollout(env, agent)

        if trial % trials_in_day == 0:
            loss = agent.smarten()
            stats.update_stats_from_agent(agent, trial, trials_in_day)
            pre_stage_transition_update()

            print(
                'Trial: {}, Action Dist:{}, Corr.:{}, Rew.:{}, loss={};'.format(stats.epoch_stats_df['Trial'].to_numpy()[-1],
                                                                                stats.epoch_stats_df['ActionDist'].to_numpy()[-1],
                                                                                stats.epoch_stats_df['Correct'].to_numpy()[-1],
                                                                                stats.epoch_stats_df['Reward'].to_numpy()[-1],
                                                                                round(loss, 2)))

            print(
                'WPI:{}, WC: {}, FC:{}'.format(stats.epoch_stats_df['WaterPreference'].to_numpy()[-1], stats.epoch_stats_df['WaterCorrect'].to_numpy()[-1],
                                               stats.epoch_stats_df['FoodCorrect'].to_numpy()[-1]))

            current_criterion = np.mean(stats.reports[-1].correct)
            reward = np.mean(stats.reports[-1].reward)
            if current_criterion > config.SUCCESS_CRITERION_THRESHOLD and reward>0.6:
                set_next_stage(env, agent)

    stats.metadata['experiment_status'] = ExperimentStatus.COMPLETED
    return stats


def set_next_stage(env: PlusMaze, agent: MotivatedAgent):
    env.set_stage(env.get_stage()+1)
    print('---------------------------------------------------------------------')
    if env.get_stage() == 1:
        env.set_random_odor_set()
        #env.set_relevant_cue(CueType.LIGHT)
        print("Stage {}: {} (Odors: {}, Correct:{})".format(env._stage, stage_names[env._stage], [np.argmax(encoding) for encoding in env.get_odor_cues()],np.argmax(env.get_correct_cue_value())))

    elif env.get_stage() == 2:
        agent.set_motivation(RewardType.FOOD)
        print("Stage {}: {}".format(env._stage, stage_names[env._stage]))

    elif env.get_stage() == 3:
        agent.set_motivation(RewardType.WATER)
        env.set_random_odor_set()
        print("Stage {}: {} (Odors: {}. Correct {})".format(env._stage, stage_names[env._stage], [np.argmax(encoding) for encoding in env.get_odor_cues()],
                                                                                 np.argmax(env.get_correct_cue_value())))
    elif env.get_stage() == 4:
        env.set_relevant_cue(CueType.LIGHT)
        print("Stage {}: {} (Lights: {}. Correct {})".format(env._stage, stage_names[env._stage], [np.argmax(encoding) for encoding in env.get_light_cues()],
                                                                                 np.argmax(env.get_correct_cue_value())))
    elif env.get_stage() == 5:
        env._relevant_cue = CueType.SPATIAL
        print("Stage {}: {} (Correct Doors: {})".format(env._stage, stage_names[env._stage], env.get_correct_cue_value()))