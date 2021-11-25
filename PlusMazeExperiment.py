from environment import PlusMaze
import numpy as np
import config
import utils
from Dashboard import Dashboard
from Stats import Stats
import os
import time

reporting_interval = 100

def PlusMazeExperiment(agent, dashboard=False):

    env = PlusMaze(relevant_cue=config.CueType.ODOR)
    stats = Stats()

    if dashboard:
        dash = Dashboard(agent.get_brain())

    def pre_stage_transition_update():
        if dashboard:
            dash.update(epoch_stats_df, env, agent.get_brain())
            dash.save_fig(dashboard_screenshots_path, env.stage)

    dashboard_screenshots_path = os.path.join('/Users/gkour/repositories/plusmaze/Results', '{}-{}'.format(agent.get_brain(),time.strftime("%Y%m%d-%H%M")))

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
                env.set_odor_options([[-1, -1], [1, 1]])
                #env.set_correct_cue_value([2])
                env.set_correct_cue_value([-1, -1])
                env.stage += 1
                print("Stage {}: Inter-dimensional shift (Odors: {}. Correct {})".format(env.stage, env._odor_options,
                                                                                         env._correct_cue_value))

                #brain.policy.controller.reset_parameters()
                pre_stage_transition_update()

            elif env.stage == 2 and current_criterion > config.SUCCESS_CRITERION_THRESHOLD:

                print("Stage 3: Transitioning to food Motivation")
                agent.set_motivation(config.RewardType.FOOD)
                env.stage += 1
                pre_stage_transition_update()
           #     brain.policy.l2.reset_parameters()
            elif env.stage == 3 and current_criterion > config.SUCCESS_CRITERION_THRESHOLD:
                print("Stage 4: Back to Water to Motivation")
                agent.set_motivation(config.RewardType.WATER)
                env.stage += 1
                #env.set_odor_options([[0.5, 0.1], [0.8, 0.3]])
                #env.set_odor_options([[3], [-3]])
                #env.set_correct_cue_value([0.5, 0.1])

                #brain.policy.controller.reset_parameters()
                pre_stage_transition_update()

            elif env.stage == 4 and current_criterion > config.SUCCESS_CRITERION_THRESHOLD:
                print("Stage 5: Extra-dimensional Shift (Light)")
                agent.set_motivation(config.RewardType.WATER)
                env.set_relevant_cue(config.CueType.LIGHT)
                env.set_odor_options([[0.02, 0.4], [0.8, 0.8]])
                #env.set_odor_options([[3], [-3]])
                env.set_correct_cue_value([0.4, 0.2])
                env.stage += 1

                #brain.policy.controller.reset_parameters()
                pre_stage_transition_update()

            elif env.stage == 5 and current_criterion > config.SUCCESS_CRITERION_THRESHOLD:
                pre_stage_transition_update()
                break

            loss_acc = 0

    return epoch_stats_df
