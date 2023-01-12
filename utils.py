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
		selection = np.random.choice(dist.nonzero()[0])
	else:
		selection = np.argmax(dist)
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


def brain_name(architecture):
	return "{}.{}".format(architecture[1].__name__, architecture[2].__name__)


def get_timestamp():
	import datetime
	return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")


def episode_rollout(env, agent):
	state = env.reset()
	# state = np.concatenate((env_state,agent.get_internal_state()))
	action_dist = agent.get_brain().think(np.expand_dims(state, 0), agent)
	action = agent.decide(state)
	dec_1hot = np.zeros(env.num_actions())
	dec_1hot[action] = 1
	new_state, outcome, terminated, info = env.step(action)
	reward = agent.evaluate_outcome(outcome)
	agent.add_experience(state, dec_1hot, reward, outcome, new_state, terminated, info)
	return state, action_dist, action, outcome, reward


def unsupervised_dimensionality(samples_embedding, explained_variance=0.95):
	num_pcs = min(len(samples_embedding), len(samples_embedding[0]))
	if num_pcs < 2:
		return np.linalg.norm(samples_embedding)
	pca = decomposition.PCA(n_components=num_pcs).fit(samples_embedding)
	dimensionality = np.cumsum(pca.explained_variance_ratio_)
	return (np.argmax(dimensionality > explained_variance) + 1)


def normalized_norm(u, ord=None):
    u = np.asarray(u)
    return np.linalg.norm(u, ord=ord)/u.size


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


def states_encoding_to_cues(states, encoding_size):
    return np.argmax(states, axis=-1) + encoding_size * np.all(states == 0, axis=-1)


def compress(a):
    return a[a != 0]