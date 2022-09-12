__author__ = 'gkour'

import os

import pandas as pd

import config
from environment import PlusMazeOneHotCues, CueType
from fitting.PlusMazeExperimentFitting import PlusMazeExperimentFitting
from fitting.fitting_config import MOTIVATED_ANIMAL_DATA_PATH
from rewardtype import RewardType
from models.networkmodels import *
from learners.networklearners import *
from learners.tabularlearners import *
from brains.consolidationbrain import ConsolidationBrain
from motivatedagent import MotivatedAgent


def test_fitting_FC_motivational():
	(nmr, lr, batch_size) = (0.8, 0.001, 20)
	env = PlusMazeOneHotCues(relevant_cue=CueType.ODOR, stimuli_encoding=10)
	rat_data = pd.read_csv(os.path.join(MOTIVATED_ANIMAL_DATA_PATH,
										"output_expr1_rat1.csv"))
	brain, learner, network = (ConsolidationBrain, DQN, FullyConnectedNetwork)
	agent = MotivatedAgent(brain(
		learner(network(env.stimuli_encoding_size(), 2, env.num_actions()), learning_rate=lr),
		batch_size=batch_size),
		motivation=RewardType.WATER,
		motivated_reward_value=config.MOTIVATED_REWARD,
		non_motivated_reward_value=nmr)

	experiment_stats, all_experiment_likelihoods = PlusMazeExperimentFitting(env, agent, dashboard=False,
																			 rat_data=rat_data)

	assert np.mean(all_experiment_likelihoods) < 1.2


if __name__ == '__main__':
	test_fitting_FC_motivational()
