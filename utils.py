__author__ = 'gkour'

import hashlib
from collections.abc import MutableMapping

import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import entropy
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


def get_inactive_doors(onehot_obs):
    encoding_size = onehot_obs.shape[-1]
    cues = stimuli_1hot_to_cues(onehot_obs, encoding_size)
    odor = cues[:, 0]  # odor for each door
    return odor == encoding_size


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def flatten_dict(d: MutableMapping, sep: str = '.') -> MutableMapping:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
    return flat_dict


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
        dist = dist / np.sum(dist)
    try:
        action = np.argmax(np.random.multinomial(1, dist))
    except:
        action = np.argmax(dist)
    return action


def softmax(x, beta=1):
    """
    Compute softmax values for each sets of scores in x.
    Rows are scores for each class.
    Columns are predictions (samples).
    """
    x_scaled = x * beta
    ex_x = np.exp(np.subtract(x_scaled, max(x_scaled)))
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
    state = env.reset_trial()
    # state = np.concatenate((env_state,agent.get_internal_state()))
    action_dist = agent.get_brain().think(np.expand_dims(state, 0), agent)
    action = agent.decide(state)
    dec_1hot = np.zeros(env.num_actions())
    dec_1hot[action] = 1
    new_state, outcome, terminated, correct_door, info = env.step(action)
    reward = agent.evaluate_outcome(outcome)
    agent.add_experience(state, dec_1hot, reward, outcome, new_state, terminated, info)
    return state, action_dist, action, outcome, reward, correct_door


def negentropy(dist, beta=1):
    return (max_entropy(len(dist)) - entropy(softmax(dist, beta))) / max_entropy(len(dist))


def max_entropy(n, beta=1):
    return entropy(softmax([1 / n] * n, beta=beta))


def unsupervised_dimensionality(samples_embedding, explained_variance=0.95):
    num_pcs = min(len(samples_embedding), len(samples_embedding[0]))
    if num_pcs < 2:
        return np.linalg.norm(samples_embedding)
    pca = decomposition.PCA(n_components=num_pcs).fit(samples_embedding)
    dimensionality = np.cumsum(pca.explained_variance_ratio_)
    return (np.argmax(dimensionality > explained_variance) + 1)


def normalized_norm(u, ord=None):
    u = np.asarray(u)
    return np.linalg.norm(u, ord=ord) / u.size


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def stimuli_1hot_to_cues(states, encoding_size):
    return np.argmax(states, axis=-1) + encoding_size * np.all(states == 0, axis=-1)


def compress(a):
    return a[a != 0]


def is_valid_attention_weights(attn):
    return all(0 <= item <= 1 for item in attn) and np.abs(np.sum(attn) - 1) < 1e-9
