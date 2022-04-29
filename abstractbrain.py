__author__ = 'gkour'

class AbstractBrain:

    def __init__(self):
        pass

    def think(self, obs, agent_state):
        '''Given an observation should return a distribution over the action set'''
        raise NotImplementedError()

    def train(self, memory, agent_state):
        raise NotImplementedError()

    def num_trainable_parameters(self):
        raise NotImplementedError()

    def get_network(self):
        raise NotImplementedError()

    def save_model(self, path):
        raise NotImplementedError()

    def load_model(self, path):
        raise NotImplementedError()

    def __str__(self):
        return self.__class__.__name__
