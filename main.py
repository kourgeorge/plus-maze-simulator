from agent import Agent
from environment import PlusMaze
import gym
import numpy as np
import config
from braindqntorch import BrainDQN
from table_dqn_brain import TableDQNBrain
from brainddpg import DDPBrain
from brainpg import BrainPG
import utils
import matplotlib.pyplot as plt
import matplotlib.animation as manimation



reporting_interval = 100
#plt.ion()
if __name__ == '__main__':
    success_criterion = 0.8

    ############ GYM env %%%%%%%%%%%%%%%%%%%%%%%%
    # env = gym.make('MountainCar-v0')
    # num_actions = env.action_space.n
    # observation_size = env.observation_space.shape[0]
    # agent = Agent(observation_size=observation_size, num_actions=num_actions)

    ############ PlusMaze env %%%%%%%%%%%%%%%%%%%%%%%%
    env = PlusMaze(relevant_cue=config.CueType.ODOR)
    num_actions = env.num_actions()
    observation_size = env.state_shape()

    #env.set_odor_options([-1, 1])
    #env.set_correct_cue_value(-1)

    # brain = BrainDQN(observation_size=observation_size, num_actions=num_actions, reward_discount=0, learning_rate=1e-4)
    # brain = TableDQNBrain(num_actions=num_actions, reward_discount=0, learning_rate=config.BASE_LEARNING_RATE)

    brain = BrainPG(observation_size, num_actions, reward_discount=0, learning_rate=config.BASE_LEARNING_RATE)
    # brain = DDPBrain(observation_size, num_actions)
    agent = Agent(brain, motivation=config.RewardType.WATER, motivated_reward_value=1, non_motivated_reward_value=0.1)

    trial = 0
    act_dist = np.zeros(num_actions)
    loss_acc = 0
    print("Stage 1: Baseline - Water Motivated, odor relevant. (Odors: {}, Correct: {})".format(env._odor_options, env._correct_cue_value))

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
    writer = FFMpegWriter(fps=1, metadata=metadata)
    fig = plt.figure(figsize=(9, 5), dpi=120, facecolor='w')
    with writer.saving(fig, "writer_test.mp4", 100):
        textstr="Stage:{}, Trial {}: Odors:{}, Lights:{}. CorrectCue: {}".format(env.stage, trial, env._odor_options, env._light_options,env._correct_cue_value)
        axis_l1 = fig.add_subplot(211)
        axis_l2 = fig.add_subplot(212)
        im1_obj = axis_l1.imshow(np.transpose(brain.policy.l1.weight.data.numpy()), cmap='RdBu', vmin=-2, vmax=2)
        im2_obj = axis_l2.imshow(brain.policy.l2.weight.data.numpy(), cmap='RdBu',  vmin=-2, vmax=2)
        props = dict(boxstyle='round', facecolor='wheat')
        figtxt = plt.figtext(0.1, 0.95, textstr, fontsize=8,verticalalignment='top', bbox=props)
        fig.colorbar(im1_obj, ax=axis_l1)
        fig.colorbar(im2_obj, ax=axis_l2)
        #plt.show()
        writer.grab_frame()

        while True:
            trial += 1
            steps, total_reward, act_dist_episode = utils.episode_rollout(env, agent)
            loss = agent.smarten()
            loss_acc += loss

            if trial % reporting_interval == 0:
                report = utils.create_report(agent.get_memory(), reporting_interval)
                print(
                    'Trial: {}, Action Dist:{}, Corr.:{}, Avg. Rew.:{}, loss={};'.format(trial,
                                    np.mean(report.action_1hot, axis=0),
                                    np.mean(report.correct),
                                    round(np.mean(report.reward), 2),
                                    round(loss_acc / reporting_interval, 2)), end='\t')

                water_preference = round(np.sum(report.arm_type_water) / len(report.arm_type_water), 2)
                water_correct_percent = round(
                    np.sum(np.logical_and(report.arm_type_water, report.correct)) /
                    np.sum(report.arm_type_water), 2)
                food_correct_percent = round(
                    np.sum(np.logical_and(np.logical_not(report.arm_type_water), report.correct)) /
                    np.sum(np.logical_not(report.arm_type_water)), 2)
                print(
                    'WPI:{}, WC: {}, FC:{}'.format(water_preference, water_correct_percent,
                                                   food_correct_percent))


                # visualize
                textstr = "Stage:{}, Trial {}: Odors:{}, Lights:{}. CorrectCue: {} \n Reward {}, Acc:{}".format(
                    env.stage, trial, env._odor_options, env._light_options, env._correct_cue_value,
                    round(np.mean(report.reward), 2),
                    round(np.mean(report.correct),2)
                    )
                figtxt.set_text(textstr)
                im1_obj.set_data(np.transpose(brain.policy.l1.weight.data.numpy()))
                im2_obj.set_data(brain.policy.l2.weight.data.numpy())
                writer.grab_frame()

                current_criterion = np.mean(report.reward)
                if env.stage == 1 and current_criterion > success_criterion:
                    env.set_odor_options([-2, 2])
                    env.set_correct_cue_value(2)
                    env.stage += 1
                    print("Stage {}: Inter-dimensional shift (Odors: {}. Correct {})".format(env.stage, env._odor_options,
                                                                                             env._correct_cue_value))



                elif env.stage == 2 and current_criterion > success_criterion:
                    print("Stage 3: Transitioning to food Motivation")
                    agent.set_motivation(config.RewardType.FOOD)
                    env.stage += 1
                elif env.stage == 3 and current_criterion > success_criterion:
                    print("Stage 4: Extra-dimensional Shift (Light)")
                    agent.set_motivation(config.RewardType.WATER)
                    env.set_relevant_cue(config.CueType.LIGHT)
                    env.set_correct_cue_value(1)
                    env.stage += 1
                elif env.stage == 4 and current_criterion > success_criterion:
                    break

                loss_acc = 0
