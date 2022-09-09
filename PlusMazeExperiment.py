__author__ = 'gkour'

from brains.consolidationbrain import ConsolidationBrain
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


trials_in_day = 100
max_experiment_length = len(PlusMazeOneHotCues.stage_names)*10 #days


class ExperimentStatus(Enum):
    COMPLETED = 'completed'
    RUNNING = 'running'


def PlusMazeExperiment(env: PlusMaze, agent:MotivatedAgent, dashboard=False):
    env.reset()
    stats = Stats(metadata={'brain': str(agent.get_brain()),
                                'network': str(agent.get_brain().get_model()),
                                'brain_params': agent.get_brain().num_trainable_parameters() if isinstance(agent.get_brain(), ConsolidationBrain) else 0,
                                'motivated_reward': agent._motivated_reward_value,
                                'non_motivated_reward': agent._non_motivated_reward_value,
                                'experiment_status': ExperimentStatus.RUNNING,
                                'stage_names': env.stage_names})

    if dashboard:
        dash = Dashboard(agent.get_brain())

    def pre_stage_transition_update():
        if dashboard:
            dash.update(stats.epoch_stats_df, env, agent.get_brain())
            dash.save_fig(dashboard_screenshots_path, env._stage)

    dashboard_screenshots_path = os.path.join('/Users/gkour/repositories/plusmaze/Results', '{}-{}'.format(agent.get_brain(),time.strftime("%Y%m%d-%H%M")))

    trial = 0
    print('============================ Brain:{}, Network:{} ======================='.format(str(agent.get_brain()), str(agent.get_brain().get_model())))
    print("Stage {}: {} - Water Motivated, odor relevant. (Odors: {}, Correct: {})".format(env.get_stage(),
                                                                                           env.stage_names[env.get_stage()],
                                                                                           [np.argmax(encoding) for encoding in env.get_odor_cues()],
                                                                                            np.argmax(env.get_correct_cue_value())))

    while env.get_stage() < len(PlusMazeOneHotCues.stage_names):

        if trial>trials_in_day*max_experiment_length:
            print("Agent failed to learn.")
            return stats

        trial += 1
        utils.episode_rollout(env, agent)
        loss = agent.smarten()
        if trial % trials_in_day == 0:
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
            if current_criterion > config.SUCCESS_CRITERION_THRESHOLD:# and reward > 0.6:
                #print(agent.get_brain().get_model().V)
                env.set_next_stage(agent)

    stats.metadata['experiment_status'] = ExperimentStatus.COMPLETED
    return stats


