__author__ = 'gkour'

import pandas as pd
from environment import PlusMaze, PlusMazeOneHotCues
from motivatedagent import MotivatedAgent
import numpy as np
#import config
from config import get_config
config = get_config()
import utils
from Dashboard import Dashboard
from Stats import Stats
import os
import time
from enum import Enum

stage_names = ['Baseline', 'IDshift', 'Mshift(Food)', 'MShift(Water)+IDshift', 'EDShift(Light)', 'EDshift(Spatial)']

trials_in_day = 100
max_experiment_length = len(stage_names)*10 #days

class EperimentStatus(Enum):
    COMPLETED = 'completed'
    RUNNING = 'running'

def PlusMazeExperiment(agent:MotivatedAgent, dashboard=False, rat_data_file=None):
    rat_data = pd.read_csv(rat_data_file) if rat_data_file is not None else None
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
    print('============================ Brain:{}, Network:{} ======================='.format(str(agent.get_brain()),str(agent.get_brain().get_network())))
    print("Stage {}: {} - Water Motivated, odor relevant. (Odors: {}, Correct: {})".format(env._stage, stage_names[env._stage], [np.argmax(encoding) for encoding in env.get_odor_cues()],
                                                                                             np.argmax(env.get_correct_cue_value())))
    likelihood_list = [] # only on real data
    while (experiment_not_finished(env, trial, rat_data)):
        if trial>trials_in_day*max_experiment_length:
            print("Agent failed to learn.")
            return stats
        
        trial += 1
                
        if (uncompleted_trial(rat_data, trial)):
            continue

        likelihood = run_rollout(env, agent, rat_data, trial)
        likelihood_list.append(likelihood)
        if day_passed(trial, trials_in_day, rat_data):
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

        if should_pass_to_next_stage(stats, rat_data, trial): #not on real data should be only when day passed?
            set_next_stage(env, agent)

    stats.metadata['experiment_status'] = EperimentStatus.COMPLETED
    return stats, np.sum(likelihood_list)


def set_next_stage(env:PlusMaze, agent:MotivatedAgent):
    env._stage += 1
    print('---------------------------------------------------------------------')
    if env._stage==1:
        env.set_random_odor_set()
        print("Stage {}: {} (Odors: {}, Correct:{})".format(env._stage, stage_names[env._stage], [np.argmax(encoding) for encoding in env.get_odor_cues()],np.argmax(env.get_correct_cue_value())))
    elif env._stage==2:
        agent.set_motivation(config.RewardType.FOOD)
        print("Stage {}: {}".format(env._stage, stage_names[env._stage]))

    elif env._stage==3:
        agent.set_motivation(config.RewardType.WATER)
        env.set_random_odor_set()
        print("Stage {}: {} (Odors: {}. Correct {})".format(env._stage, stage_names[env._stage], [np.argmax(encoding) for encoding in env.get_odor_cues()],
                                                                                 np.argmax(env.get_correct_cue_value())))
    elif env._stage==4:
        env.set_relevant_cue(config.CueType.LIGHT)
        print("Stage {}: {} (Lights: {}. Correct {})".format(env._stage, stage_names[env._stage], [np.argmax(encoding) for encoding in env.get_light_cues()],
                                                                                 np.argmax(env.get_correct_cue_value())))

    elif env._stage==5:
        env._relevant_cue = config.CueType.SPATIAL
        print("Stage {}: {} (Correct Doors: {})".format(env._stage, stage_names[env._stage], env.get_correct_cue_value()))


def experiment_not_finished(env:PlusMaze, trial, rat_data):
    if rat_data is not None:
        if trial < len(rat_data):
            return True
        else:
            return False
    else:
        env._stage < len(stage_names)

def should_pass_to_next_stage(stats, rat_data, trial):
    if rat_data is not None:
        return  (trial < len(rat_data) and rat_data.iloc[trial]['stage'] > rat_data.iloc[trial-1]['stage'])
    else:
        current_criterion = np.mean(stats.reports[-1].correct)
        reward = np.mean(stats.reports[-1].reward)
        return (current_criterion > config.SUCCESS_CRITERION_THRESHOLD and reward>0.6)

def day_passed(trial, trials_in_day, rat_data):
    if rat_data is not None:
        return (trial < len(rat_data) and rat_data.iloc[trial]['day in stage'] != rat_data.iloc[trial-1]['day in stage']) #change to day_in_stage
    else:
        return (trial % trials_in_day == 0)

def uncompleted_trial(rat_data, trial): 
    return (rat_data.iloc[trial-1].reward_type == 0) #should be NaN)


def run_rollout(env, agent, rat_data, trial):
    if rat_data is not None:
        _,_,_, likelihood =  utils.episode_rollout_on_real_data(env, agent, rat_data.iloc[trial - 1])
        return likelihood
    else:
        utils.episode_rollout(env, agent)