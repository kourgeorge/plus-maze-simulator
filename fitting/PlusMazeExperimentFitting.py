__author__ = 'gkour'

import numpy as np
import os
import time
import config
from PlusMazeExperiment import ExperimentStatus
from environment import PlusMazeOneHotCues, CueType, PlusMaze
from motivatedagent import MotivatedAgent
import fitting.fitting_utils as fitting_utils
from Dashboard import Dashboard
from fitting.FittingStats import FittingStats


def PlusMazeExperimentFitting(env: PlusMaze, agent: MotivatedAgent, experiment_data, dashboard=False):

    fitting_info = experiment_data.copy()
    fitting_info['model_reward'] = np.nan
    fitting_info['model_action_dist'] = np.nan
    fitting_info['model_action'] = np.nan
    fitting_info['model_action_dist'] = fitting_info['model_action_dist'].astype(object)
    fitting_info['likelihood'] = np.nan
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
    print("Stage {}: {} - Water Motivated, odor relevant. (Odors: {}, Correct: {})".format(env.get_stage(), env.stage_names[env.get_stage()],
                                                                                           [np.argmax(encoding) for encoding in env.get_odor_cues()],
                                                                                             np.argmax(env.get_correct_cue_value())))
    model_action_dists = np.empty([1, env.num_actions()])
    while trial < len(fitting_info):
        if completed_trial(fitting_info, trial):
            _, _, _, model_action_dist, model_action, likelihood, model_action_outcome = fitting_utils.episode_rollout_on_real_data(env, agent,
                                                                                                                      fitting_info.iloc[trial])
            model_action_dists = np.append(model_action_dists, np.expand_dims(model_action_dist, axis=0), axis=0)

            fitting_info['likelihood'].loc[trial] = likelihood
            fitting_info['model_action_dist'].loc[trial] = np.round(model_action_dist,3)
            fitting_info['model_action'].loc[trial] = model_action
            fitting_info['model_reward'].loc[trial] = agent.evaluate_outcome(model_action_outcome)

            loss = agent.smarten()

        if day_passed(trial, fitting_info):
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
        if should_pass_to_next_stage(fitting_info, trial):
            env.set_next_stage(agent)

    print("Likelihood - Average:{}, Median:{}".format(np.mean(fitting_info.likelihood), np.median(fitting_info.likelihood)))
    stats.metadata['experiment_status'] = ExperimentStatus.COMPLETED
    return stats, fitting_info


def should_pass_to_next_stage(rat_data, trial):
    return trial < len(rat_data) and rat_data.iloc[trial]['stage'] > rat_data.iloc[trial - 1]['stage']


def day_passed(trial, rat_data):
    return 0 < trial < len(rat_data) and \
           (rat_data.iloc[trial]['day in stage'] != rat_data.iloc[trial - 1]['day in stage'] or  # day change
             rat_data.iloc[trial]['stage'] != rat_data.iloc[trial - 1]['stage'])  # stage change


def completed_trial(rat_data, trial):
    return not np.isnan(rat_data.iloc[trial].action) and not rat_data.iloc[trial]["A{}o".format(int(rat_data.iloc[trial].action))]==-1
