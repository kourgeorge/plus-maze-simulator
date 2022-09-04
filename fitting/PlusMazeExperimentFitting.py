__author__ = 'gkour'

import numpy as np
import os
import time
import config
from environment import PlusMazeOneHotCues, CueType
from PlusMazeExperiment import stage_names, ExperimentStatus, set_next_stage
from motivatedagent import MotivatedAgent
import fitting_utils as fitting_utils
from Dashboard import Dashboard
from FittingStats import FittingStats


def PlusMazeExperimentFitting(agent: MotivatedAgent, rat_data, dashboard=False):

    env = PlusMazeOneHotCues(relevant_cue=CueType.ODOR)
    env.reset()
    stats = FittingStats(metadata={'brain': str(agent.get_brain()),
                                'network': str(agent.get_brain().get_model()),
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
    print('============================ Brain:{}, Network:{} ======================='.format(str(agent.get_brain()), str(agent.get_brain().get_model())))
    print("Stage {}: {} - Water Motivated, odor relevant. (Odors: {}, Correct: {})".format(env._stage, stage_names[env._stage], [np.argmax(encoding) for encoding in env.get_odor_cues()],
                                                                                             np.argmax(env.get_correct_cue_value())))
    likelihood_list = []  # only on real data
    model_action_dists = np.empty([1, env.num_actions()])
    while trial < len(rat_data):
        if completed_trial(rat_data, trial):
            _, _, _, model_action_dist, likelihood = fitting_utils.episode_rollout_on_real_data(env, agent,
                                                                                                rat_data.iloc[trial])
            likelihood_list.append(likelihood)
            model_action_dists = np.append(model_action_dists, np.expand_dims(model_action_dist, axis=0), axis=0)
            loss = agent.smarten()

        if day_passed(trial, rat_data):
            stats.update_stats_from_agent(agent, trial, config.REPORTING_INTERVAL)
            pre_stage_transition_update()

            print(
                'Trial: {}, Action Dist:{}, Model Dist:{}, Corr.:{}, Rew.:{}, loss={}, likelihood:{}'.format(stats.epoch_stats_df['Trial'].to_numpy()[-1],
                                                                                stats.epoch_stats_df['ActionDist'].to_numpy()[-1],
                                                                                np.round(np.mean(model_action_dists, axis=0),2),
                                                                                stats.epoch_stats_df['Correct'].to_numpy()[-1],
                                                                                stats.epoch_stats_df['Reward'].to_numpy()[-1],
                                                                                round(loss, 2),
                                                                                round(stats.epoch_stats_df['Likelihood'].to_numpy()[-1],3)))

            print(
                'WPI:{}, WC: {}, FC:{}'.format(stats.epoch_stats_df['WaterPreference'].to_numpy()[-1], stats.epoch_stats_df['WaterCorrect'].to_numpy()[-1],
                                                stats.epoch_stats_df['FoodCorrect'].to_numpy()[-1]))
            model_action_dists = np.empty([1, env.num_actions()])

        trial += 1
        if should_pass_to_next_stage(rat_data, trial):
            set_next_stage(env, agent)

    print("Likelihood - Average:{}, Median:{}".format(np.mean(likelihood_list), np.median(likelihood_list)))
    stats.metadata['experiment_status'] = ExperimentStatus.COMPLETED
    return stats, likelihood_list


def should_pass_to_next_stage(rat_data, trial):
    return trial < len(rat_data) and rat_data.iloc[trial]['stage'] > rat_data.iloc[trial - 1]['stage']


def day_passed(trial, rat_data):
    return 0 < trial < len(rat_data) and \
           (rat_data.iloc[trial]['day in stage'] != rat_data.iloc[trial - 1]['day in stage'] or  # day change
             rat_data.iloc[trial]['stage'] != rat_data.iloc[trial - 1]['stage'])  # stage change


def completed_trial(rat_data, trial):
    return rat_data.iloc[trial].reward_type != 0
