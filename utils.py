import numpy as np


def epsilon_greedy(eps, dist):
    p = np.random.rand()
    if p < eps:
        selection = np.random.randint(low=0, high=len(dist))
    else:
        selection = np.argmax(dist)

    return selection

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