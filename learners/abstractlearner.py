__author__ = 'gkour'


class AbstractLearner:
	def __init__(self, model, optimizer):
		super().__init__()
		self.model = model
		self.optimizer = optimizer

	def get_model(self):
		return self.model

	def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation):
		raise NotImplementedError()

	def __str__(self):
		return  self.__class__.__name__
