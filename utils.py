import numpy as np
from sklearn import decomposition
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
        new_state, reward, terminated, info = env.step(action)
        reward = agent.evaluate_reward(reward)
        total_reward += reward
        agent.add_experience(state, dec_1hot, reward, new_state, terminated, info)
        state = new_state

    return steps, total_reward, act_dist



def electrophysiology_analysis(brain:AbstractBrain):
    affine = brain.network.get_stimuli_layer().T.detach().numpy()
    affine_dim = unsupervised_dimensionality(affine)
    controller = brain.network.get_door_attention().T.detach().numpy()
    controller_dim = unsupervised_dimensionality(controller)

    return affine_dim, controller_dim


def unsupervised_dimensionality(samples_embedding, explained_variance=0.95):
    return 1
    num_pcs = min(len(samples_embedding), len(samples_embedding[0]))
    pca = decomposition.PCA(n_components=num_pcs).fit(samples_embedding)
    dimensionality = np.cumsum(pca.explained_variance_ratio_)
    return (np.argmax(dimensionality > explained_variance) + 1)
