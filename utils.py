import numpy as np
from sklearn import decomposition

import config
from abstractbrain import AbstractBrain


class Object(object):
    pass

def has_err(x):
    return bool(((x != x) | (x == float("inf")) | (x == float("-inf"))).any().item())

def epsilon_greedy(eps, dist):
    p = np.random.rand()
    if p < eps:
        selection = np.random.randint(low=0, high=len(dist))
    else:
        selection = dist_selection(dist)
        #selection = np.argmax(dist)
    return selection


def dist_selection(dist):
    # dist = softmax(dist)
    # select_prob = np.random.choice(dist, p=dist)
    # selection = np.argmax(dist == select_prob)
    if sum(dist) != 1:
        dist = dist/np.sum(dist)
    try:
        action = np.argmax(np.random.multinomial(1, dist))
    except:
        action = np.argmax(dist)
    return action


def softmax(x, temprature=1):
    """
    Compute softmax values for each sets of scores in x.

    Rows are scores for each class.
    Columns are predictions (samples).
    """
    # x = normalize(np.reshape(x, (1, -1)), norm='l2')[0]
    ex_x = np.exp(temprature * np.subtract(x, max(x)))
    if np.isinf(np.sum(ex_x)):
        raise Exception('Inf in softmax')
    return ex_x / ex_x.sum(0)


def dot_lists(V1, V2):
    return sum([x * y for x, y in zip(V1, V2)])



def episode_rollout(env, agent):
    total_reward = 0

    num_actions = env.num_actions()

    act_dist = np.zeros(num_actions)

    env_state = env.reset()
    terminated = False
    steps = 0
    while not terminated:
        steps += 1
        # env.render()
        #state = np.concatenate((env_state,agent.get_internal_state()))
        state = env_state
        action = agent.decide(state)
        dec_1hot = np.zeros(num_actions)
        dec_1hot[action] = 1
        act_dist += dec_1hot
        new_state, reward, terminated, info = env.step(action)
        reward = agent.evaluate_reward(reward)
        total_reward += reward
        agent.add_experience(state, dec_1hot, reward, new_state, terminated, info)
        state = new_state

    return steps, total_reward, act_dist


def create_report_from_memory(experience: ReplayMemory, brain:AbstractBrain, last):
    if last == -1:
        pass
    last_exp = experience.last(last)
    actions = [data[1] for data in last_exp]
    rewards = [data[2] for data in last_exp]
    infos = [data[5] for data in last_exp]

    report_dict = Object()
    report_dict.action_1hot = actions
    report_dict.action = np.argmax(actions, axis=1)
    report_dict.reward = rewards
    report_dict.arm_type_water = [1 if action < 2 else 0 for action in report_dict.action]
    report_dict.correct = [1 if info.outcome != config.RewardType.NONE else 0 for info in infos]
    report_dict.stage = [info.stage for info in infos]
    report_dict.affine_dim = electrophysiology_analysis(brain)
    return report_dict


def electrophysiology_analysis(brain:AbstractBrain):
    affine = brain.network.affine.weight.T.detach().numpy()
    affine_dim = unsupervised_dimensionality(affine)
    return affine_dim


def unsupervised_dimensionality(samples_embedding, explained_variance=0.95):
    num_pcs = min(len(samples_embedding), len(samples_embedding[0]))
    pca = decomposition.PCA(n_components=num_pcs).fit(samples_embedding)
    dimensionality = np.cumsum(pca.explained_variance_ratio_)
    return (np.argmax(dimensionality > explained_variance) + 1)
