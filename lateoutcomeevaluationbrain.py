__author__ = 'gkour'

import numpy as np
import torch
from abstractbrain import AbstractBrain


class LateOutcomeEvaluationBrain(AbstractBrain):
	BATCH_SIZE = 20

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def consolidate(self, memory, agent, batch_size=BATCH_SIZE, iterations=100):
		minibatch_size = min(batch_size, len(memory))
		if minibatch_size == 0:
			return
		losses = []
		for _ in range(iterations):
			minibatch = memory.sample(minibatch_size)
			state_batch = torch.from_numpy(np.stack([np.stack(data[0]) for data in minibatch])).float()
			action_batch = torch.FloatTensor([data[1] for data in minibatch])
			outcome_batch = [data[3] for data in minibatch]
			reward_batch = torch.FloatTensor([agent.evaluate_outcome(outcome) for outcome in outcome_batch])
			nextstate_batch = torch.from_numpy(np.stack([data[4] for data in minibatch])).float()

			action_values = self.think(state_batch, agent)

			losses += [self.optimize(state_batch, action_batch, reward_batch, action_values, nextstate_batch)]

		return np.mean(losses)