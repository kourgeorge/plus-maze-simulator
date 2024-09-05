__author__ = 'gkour'

import json

import pandas as pd

from brains.consolidationbrain import ConsolidationBrain
from environment import PlusMaze, PlusMazeOneHotCues, CueType
from models.tabularmodels import FTable
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


class ExperimentStatus(Enum):
    COMPLETED = 'completed'
    RUNNING = 'running'


def PlusMazeExperiment(env: PlusMaze, agent: MotivatedAgent, dashboard=False):
    max_experiment_length = len(env.stage_names) * 10  # days
    env.init()
    experiment_data = pd.DataFrame()
    stats = Stats(metadata={'brain': str(agent.get_brain()),
                                'learner': str(agent.get_brain().get_learner()),
                                'model': str(agent.get_brain().get_model()),
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
    day_in_stage = 1
    print('============================ Brain:{}, Learner:{}, Model:{}, ======================='.format(str(agent.get_brain()), str(agent.get_brain().get_learner()), str(agent.get_brain().get_model())))
    print(f"Stage {env.get_stage()}: {env.stage_names[env.get_stage()]} - Water Motivated."
          f"(Odors: {[np.argmax(encoding) for encoding in env.get_odor_cues()]}, Correct: {np.argmax(env.get_correct_cue_value())})")

    while env.get_stage() < len(env.stage_names):

        if trial > config.TRIALS_IN_DAY * max_experiment_length:
            print("Agent failed to learn.")
            return stats, experiment_data

        trial += 1
        state, action_dist, action, outcome, reward, correct_door = utils.episode_rollout(env, agent)
        trial_dict = env.format_state(state)

        trial_dict['trial'] = trial % config.TRIALS_IN_DAY
        trial_dict['stage'] = env.get_stage() + 1
        trial_dict['stage_name'] = env.stage_names[env.get_stage()]
        trial_dict['action'] = action + 1
        trial_dict['reward'] = 0 if outcome == RewardType.NONE else 1
        trial_dict['day in stage'] = day_in_stage
        trial_dict['model_variables'] = agent.get_brain().get_model().get_model_metrics()
        #experiment_data = experiment_data.append(trial_dict, ignore_index=True)
        optimization_data = agent.smarten()
        trial_dict['optimization_data'] = json.dumps(optimization_data)
        trial_dict['model'] = json.dumps(optimization_data)
        trial_dict['model_action_dist'] = agent.get_brain().think(np.expand_dims(state, 0), agent).squeeze().detach().numpy()
        trial_dict['correct_door'] = correct_door
        experiment_data = pd.concat([experiment_data, pd.DataFrame.from_records([trial_dict])])

        if trial % config.TRIALS_IN_DAY == 0:
            day_in_stage += 1
            stats.update_stats_from_agent(agent, trial, config.TRIALS_IN_DAY)
            pre_stage_transition_update()

            print(
                'Trial: {}, Action Dist:{}, Corr.:{}, Rew.:{}, loss={};'.format(stats.epoch_stats_df['Trial'].to_numpy()[-1],
                                                                                stats.epoch_stats_df['ActionDist'].to_numpy()[-1],
                                                                                stats.epoch_stats_df['Correct'].to_numpy()[-1],
                                                                                stats.epoch_stats_df['Reward'].to_numpy()[-1],
                                                                                round(optimization_data['delta'], 2)))

            print(
                'WPI:{}, WC: {}, FC:{}'.format(stats.epoch_stats_df['WaterPreference'].to_numpy()[-1], stats.epoch_stats_df['WaterCorrect'].to_numpy()[-1],
                                               stats.epoch_stats_df['FoodCorrect'].to_numpy()[-1]))

            current_criterion = stats.reports[-1].reward
            reward = stats.reports[-1].reward

            if reward != experiment_data.tail(config.TRIALS_IN_DAY)['reward'].mean():
                raise Exception()
            if current_criterion > config.SUCCESS_CRITERION_THRESHOLD:# and reward > 0.6:
                day_in_stage = 1
                if hasattr(agent.get_brain().get_model(), 'attn_importance'):
                    print(agent.get_brain().get_model().attn_importance)

                #print(torch.softmax(agent.get_brain().get_model().attn_importance, axis=0))
                env.set_next_stage(agent)

    stats.metadata['experiment_status'] = ExperimentStatus.COMPLETED
    return stats, experiment_data


