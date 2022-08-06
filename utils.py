__author__ = 'gkour'

import hashlib

import numpy as np
from matplotlib import cm
from sklearn import decomposition


class Object(object):
    pass

def has_err(x):
    return bool(((x != x) | (x == float("inf")) | (x == float("-inf"))).any().item())

def colorify(name):
	h = int(hashlib.sha1(bytes(name, 'ascii')).hexdigest(), 16)
	return [h % 100 / 100,
			h / 100 % 100 / 100,
			h / 10000 % 100 / 100]

def colorify2(name):
	colors = cm.rainbow(np.linspace(0, 1, 1000))
	h = int(hashlib.sha1(bytes(name, 'ascii')).hexdigest(), 16)
	return colors[h % 1000]

def epsilon_greedy(eps, dist):
    p = np.random.rand()
    if p < eps:
        selection = np.random.randint(low=0, high=len(dist))
    else:
        selection = dist_selection(dist)
    return selection


def dist_selection(dist):
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
        new_state, outcome, terminated, info = env.step(action)
        reward = agent.evaluate_outcome(outcome)
        total_reward += reward
        agent.add_experience(state, dec_1hot, reward, outcome, new_state, terminated, info)
        state = new_state

    return steps, total_reward, act_dist

def episode_rollout_on_real_data(env, agent, current_trial):
    total_reward = 0

    num_actions = env.num_actions()

    act_dist = np.zeros(num_actions)

    env_state = env.reset()
    terminated = False
    steps = 0
    likelihood = 0
    while not terminated:
        # print("trial: {}".format(current_trial['trial']))
        # print("rewarded in real data: {}, from type: {}".format(current_trial['reward'], current_trial['reward_type']))
        steps += 1
        state = env.set_state(current_trial)
        action = int(current_trial['action']) - 1
        dec_1hot = np.zeros(num_actions)
        dec_1hot[action] = 1
        act_dist += dec_1hot
        # print_state(state, action, env)
        new_state, outcome, terminated, info = env.step(action)
        # print ("outcome: ", outcome)
        reward = agent.evaluate_outcome(outcome)
        total_reward += reward
        model_action_dist=agent._brain.think(np.expand_dims(state,0), agent).squeeze().detach().numpy()
        likelihood += -1 * np.log(model_action_dist[action])

        agent.add_experience(state, dec_1hot, reward, outcome, new_state, terminated, info)

        state = env.set_state(current_trial)
        info.likelihood = likelihood
        info.network_action = agent.decide(state)
        _, outcome_network, _, _ = env.step(info.network_action)
        info.network_outcome = outcome_network

        state = new_state
    return steps, total_reward, act_dist, model_action_dist, likelihood



def unsupervised_dimensionality(samples_embedding, explained_variance=0.95):
    num_pcs = min(len(samples_embedding), len(samples_embedding[0]))
    pca = decomposition.PCA(n_components=num_pcs).fit(samples_embedding)
    dimensionality = np.cumsum(pca.explained_variance_ratio_)
    return (np.argmax(dimensionality > explained_variance) + 1)


def normalized_norm(u, ord=None):
    u = np.asarray(u)
    return np.linalg.norm(u, ord=ord)/u.size

def print_state(state, action, env):
    odors = state[0, :, :]
    lights = state[1, :, :]
    print ("correct odor: {}".format(np.argmax(env.get_correct_cue_value())))
    for i in range(len(odors)):
        print(i, np.argmax(odors[i]), np.argmax(lights[i]))
    print ("action: {}".format(action))
    # print ("rewarded in simulation: {}".format(outcome))