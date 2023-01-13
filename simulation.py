__author__ = 'gkour'

import numpy as np
import pandas as pd

from brains.lateoutcomeevaluationbrain import LateOutcomeEvaluationBrain
from motivatedagent import MotivatedAgent
from environment import PlusMazeOneHotCues2ActiveDoors, CueType, PlusMazeOneHotCues

import os
import config
from models.networkmodels import *
from learners.networklearners import *
from learners.tabularlearners import *
from brains.fixeddoorattentionbrain import FixedDoorAttentionBrain
from brains.motivationdependantbrain import MotivationDependantBrain
from PlusMazeExperiment import PlusMazeExperiment, ExperimentStatus
from behavioral_analysis import plot_days_per_stage, plot_behavior_results
from brains.consolidationbrain import ConsolidationBrain
from rewardtype import RewardType
from brains.tdbrain import TDBrain

brains = [  # ((TDBrain, QLearner, QTable), (4.5, 0.032)),
	# ((TDBrain, OptionsLearner, OptionsTable), (4.6, 0.0087)),
	# ((TDBrain, IALearner, ACFTable), (6.3, 0.0065)),
	# ((TDBrain, IAAluisiLearner, ACFTable), (6.0, 0.0036)),
	# ((TDBrain, MALearner, ACFTable), (5.9, 0.0094, 0.051)),
	# # (TDBrain, MALearnerSimple, ACFTable),
	((ConsolidationBrain, DQN, UANet), (config.BETA, config.LEARNING_RATE)),
	((ConsolidationBrain, DQN, ACLNet), (config.BETA, config.LEARNING_RATE)),
	((ConsolidationBrain, DQN, FCNet), (config.BETA, config.LEARNING_RATE)),
	((ConsolidationBrain, DQN, FC2LayersNet), (config.BETA, config.LEARNING_RATE)),
	#((ConsolidationBrain, DQN, EfficientNetwork), (config.BETA, config.LEARNING_RATE)),
	# (FixedDoorAttentionBrain, DQN, EfficientNetwork),
	# (MotivationDependantBrain, DQN, SeparateMotivationAreasNetwork),
	# (MotivationDependantBrain, DQN, SeparateMotivationAreasFCNetwork),
	# (LateOutcomeEvaluationBrain, DQN, SeparateMotivationAreasNetwork)
]


def sample_from_estimated_parameters(model_name):
	ranges = [(0, 30), (0.001, 0.2), (0.001, 0.2)]
	estimated_params_values = {'QLearner.QTable': {'beta': [4.5, 0.33], 'alpha': [0.032, 0.011]},
							   'OptionsLearner.OptionsTable': {'beta': [4.6, 0.35], 'alpha': [0.0087, 0.0014]},
							   'IALearner.ACFTable': {'beta': [6.3, 0.86], 'alpha': [0.0065, 0.00068]},
							   'IAAluisiLearner.ACFTable': {'beta': [6.0, 0.8], 'alpha': [0.0036, 0.00068]},
							   'MALearner.ACFTable': {'beta': [5.9, 0.97], 'alpha': [0.0094, 0.0016],
													  'alpha_phi': [0.051, 0.018]}}

	parameters = estimated_params_values[model_name]
	sampled_simulation_params = [np.random.normal(loc=mean, scale=sem * np.sqrt(8)) for mean, sem in
								 parameters.values()]
	return tuple(np.clip(sampled_simulation_params[i], *ranges[i]) for i, parameter_value in
				 enumerate(sampled_simulation_params))


def run_simulation(env):
	repetitions = 20

	all_experiment_data = pd.DataFrame()
	brains_reports = []
	for agent_spec in brains:
		completed_experiments = 0
		aborted_experiments = 0
		brain_repetition_reports = [None] * repetitions
		while completed_experiments < repetitions:
			env.init()

			architecture, parameters = agent_spec
			estimated_parameters = parameters
			#estimated_parameters = sample_from_estimated_parameters(utils.brain_name(architecture))


			(brain, learner, model) = architecture
			model_instance = model(env.stimuli_encoding_size(), 2, env.num_actions())

			if issubclass(learner, MALearner) or issubclass(learner, DQNAtt):
				(beta, lr, attention_lr) = estimated_parameters
				learner_instance = learner(model_instance, learning_rate=lr, alpha_phi=attention_lr)
			else:
				(beta, lr) = estimated_parameters
				learner_instance = learner(model_instance, learning_rate=lr)

			agent = MotivatedAgent(brain(learner_instance, beta=beta),
								   motivation=RewardType.WATER, motivated_reward_value=config.MOTIVATED_REWARD,
								   non_motivated_reward_value=config.NON_MOTIVATED_REWARD)
			experiment_stats, experiment_data = PlusMazeExperiment(env, agent, dashboard=False)
			experiment_data['model'] = utils.brain_name(architecture)
			experiment_data['subject'] = completed_experiments

			if experiment_stats.metadata['experiment_status'] == ExperimentStatus.COMPLETED:
				brain_repetition_reports[completed_experiments] = experiment_stats
				completed_experiments += 1
				all_experiment_data = pd.concat([all_experiment_data,experiment_data])
			else:
				aborted_experiments += 1
		brains_reports.append(brain_repetition_reports)
		print("{} out of {} experiments were aborted".format(aborted_experiments,
															 aborted_experiments + completed_experiments))

	all_experiment_data.to_csv(
		'/Users/gkour/repositories/plusmaze/fitting/Results/simulations_results/simulation_{}_{}.csv'.format(
			repetitions, utils.get_timestamp()))

	plot_days_per_stage(brains_reports, file_path=os.path.join('Results', 'days_per_stage.png'))

	for brain_report in brains_reports:
		plot_behavior_results(brain_report)


if __name__ == '__main__':
	# env = PlusMazeOneHotCues2ActiveDoors(relevant_cue=CueType.ODOR, stimuli_encoding=8)
	env = PlusMazeOneHotCues(relevant_cue=CueType.ODOR, stimuli_encoding=10)
	run_simulation(env)
	x = 1
