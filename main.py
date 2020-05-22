from agent import Agent
from environment import PlusMaze
import numpy as np
import config

if __name__ == '__main__':

    reporting_interval = 1000

    agent = Agent(motivation=config.RewardType.WATER, motivated_reward_value=1, non_motivated_reward_value=1)
    env = PlusMaze(relevant_cue=config.CueType.ODOR, correct_value=1)
    env.reset()
    state = env.state()

    act_dist = np.zeros(4)
    reward_acc = 0
    for trial in range(1, 60000):
        action = agent.decide(env.state())
        dec_1hot = np.zeros(len(env.action_space()))
        dec_1hot[action] = 1
        act_dist += dec_1hot
        outcome, new_state, terminated = env.step(action)
        reward = agent.evaluate_reward(outcome)
        reward_acc += reward
        #print('Action:{}, Reward:{}'.format(action, reward))
        agent.add_experience([state, dec_1hot, reward, new_state, terminated])
        state = new_state

        if trial % config.BASE_LEARNING_FREQ == 0:
            agent.smarten()

        if trial % reporting_interval == 0:
            print('Action:{}, Reward:{}'.format(act_dist / reporting_interval, reward_acc / reporting_interval))
            act_dist = np.zeros(4)
            reward_acc = 0

        # if trial == 35000:
        #     print("Stage 2: change the rewarded value")
        #     env = PlusMaze(relevant_cue=config.CueType.ODOR, correct_value=-1)
        # if trial == 45000:
        #     print("Stage 3: Transitioning to food Motivation")
        #     agent.set_motivation(config.RewardType.FOOD)
