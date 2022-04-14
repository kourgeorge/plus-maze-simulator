from environment import PlusMaze
from motivatedagent import MotivatedAgent
import numpy as np
import config
import utils
from Dashboard import Dashboard
from Stats import Stats
import os
import time

reporting_interval = 100


def PlusMazeExperiment(agent:MotivatedAgent, dashboard=False):

    env = PlusMaze(relevant_cue=config.CueType.ODOR)
    env.reset()
    stats = Stats()

    if dashboard:
        dash = Dashboard(agent.get_brain())

    def pre_stage_transition_update():
        if dashboard:
            dash.update(epoch_stats_df, env, agent.get_brain())
            dash.save_fig(dashboard_screenshots_path, env._stage)

    dashboard_screenshots_path = os.path.join('/Users/gkour/repositories/plusmaze/Results', '{}-{}'.format(agent.get_brain(),time.strftime("%Y%m%d-%H%M")))

    trial = 0
    loss_acc = 0
    print('============================ Brain:{} ======================='.format(str(agent.get_brain())))
    print("Stage 0: Baseline - Water Motivated, odor relevant. (Odors: {}, Correct: {})".format(env.get_odor_cues(),
                                                                                                env.get_correct_cue_value()))
    while env._stage < 5:
        trial += 1
        utils.episode_rollout(env, agent)
        loss = agent.smarten()
        loss_acc += loss
        if trial % reporting_interval == 0:

            report = utils.create_report_from_memory(agent.get_memory(), agent.get_brain(), reporting_interval)
            epoch_stats_df = stats.update(trial, report)
            pre_stage_transition_update()

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
            if current_criterion > config.SUCCESS_CRITERION_THRESHOLD:
                set_next_stage(env,agent)

            loss_acc = 0

    epoch_stats_df._metadata = {'brain': str(agent.get_brain()),
                                'brain_params': agent.get_brain().num_trainable_parameters(),
                                'motivated_reward': agent._motivated_reward_value,
                                'non_motivated_reward': agent._non_motivated_reward_value}
    return epoch_stats_df


def set_next_stage(env:PlusMaze, agent:MotivatedAgent):
    env._stage += 1
    print('---------------------------------------------------------------------')
    if env._stage==1:
        env.set_random_odor_set()
        print("Stage {}: IDshift (Odors: {}. Correct {})".format(env._stage, env.get_odor_cues(),
                                                                                 env.get_correct_cue_value()))
    elif env._stage==2:
        agent.set_motivation(config.RewardType.FOOD)
        print("Stage 2: Mshift{}".format(agent.get_motivation()))

    elif env._stage==3:
        agent.set_motivation(config.RewardType.WATER)
        env.set_random_odor_set()
        print("Stage 3: MShift(Water)+IDshift (Odors: {}. Correct {})".format(env._stage, env.get_odor_cues(),
                                                                                 env.get_correct_cue_value()))
    elif env._stage==4:
        env.set_relevant_cue(config.CueType.LIGHT)
        print("Stage 4: EDShift (Light).")