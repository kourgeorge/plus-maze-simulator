__author__ = 'gkour'

import pandas as pd

from PlusMazeExperiment import PlusMazeExperiment, ExperimentStatus
from behavioral_analysis import plot_days_per_stage, plot_behavior_results
from brains.consolidationbrain import ConsolidationBrain
from brains.tdbrain import TDBrain
from environment import CueType, PlusMazeOneHotCues, PlusMazeOneHotCues2ActiveDoors
from fitting.fitting_utils import extract_names_from_architecture, string2list
from learners.networklearners import *
from learners.tabularlearners import *
from models.networkmodels import *
from models.tabularmodels import *
from motivatedagent import MotivatedAgent
from rewardtype import RewardType

brains = [  # ((TDBrain, QLearner, QTable), (4.5, 0.032)),
	# ((TDBrain, OptionsLearner, OptionsTable), (4.6, 0.0087)),
	((TDBrain, IALearner, FTable), (2, 0.26)),
	# ((TDBrain, IAAluisiLearner, ACFTable), (6.0, 0.0036)),
	#((TDBrain, MALearner, ACFTable), (5.9, 0.0094, 0.051)),
	# # (TDBrain, MALearnerSimple, ACFTable),
	#((ConsolidationBrain, DQN, FC2LayersNet), (3, 0.26)),
	#((ConsolidationBrain, DQN, ACLNet), (config.BETA, config.LEARNING_RATE)),
	((ConsolidationBrain, DQN, UANet), (2, 0.26)),
	#((ConsolidationBrain, DQN, FCNet), (0.5, 0.079)),
	# ((ConsolidationBrain, DQN, FC2LayersNet), (config.BETA, config.LEARNING_RATE)),
	#((ConsolidationBrain, DQN, EfficientNetwork), (config.BETA, config.LEARNING_RATE)),
	# (FixedDoorAttentionBrain, DQN, EfficientNetwork),
	# (MotivationDependantBrain, DQN, SeparateMotivationAreasNetwork),
	# (MotivationDependantBrain, DQN, SeparateMotivationAreasFCNetwork),
	# (LateOutcomeEvaluationBrain, DQN, SeparateMotivationAreasNetwork)
]

brains=[
	#((TDBrain, QLearner, QTable), (config.BETA, config.LEARNING_RATE)),
	 # ((TDBrain, QLearner, MQTable), (config.BETA, config.LEARNING_RATE)),
	# ((TDBrain, UABQLearner, QTable), (config.BETA, config.LEARNING_RATE)),
	#((TDBrain, ABQLearner, QTable), (config.BETA, config.LEARNING_RATE)),

	#((TDBrain, QLearner, OptionsTable), (config.BETA, config.LEARNING_RATE)),
	#((TDBrain, QLearner, MOptionsTable), (config.BETA, config.LEARNING_RATE)),
	# ((TDBrain, UABQLearner, OptionsTable), (config.BETA, config.LEARNING_RATE)),
	# ((TDBrain, ABQLearner, OptionsTable), (config.BETA, config.LEARNING_RATE)),
	#((TDBrain, IALearner, FTable), (config.BETA, config.LEARNING_RATE)),
	# ((TDBrain, ABIALearner, SCFDependantV), (config.BETA, config.LEARNING_RATE)),

	#((TDBrain, ABIALearner, FTable), (config.BETA, config.LEARNING_RATE)),
	#((TDBrain, ABIALearner, MFTable), (-0.1, 3, 0.005, 0.01)), #M(VB)
	#((TDBrain, ABIALearner, MSCFTable), (0, 3, 0.008, 0.01)), #M(VB)+S(V)
((TDBrain, ABIALearner, MSCFTable), (0, 4, 0.004, 0.008)), #M(VB)+S(V)
	#((TDBrain, ABIALearner, SCVBFTable), (0.0, 3,0.010, 0.01)),

]

def sample_from_estimated_parameters(model_name):
	ranges = [(-1,1), (1, 30), (0.001, 0.4), (0.001, 0.4)]
	estimated_params_values = {
		'OptionsLearner.OptionsTable': {'beta': [4.6, 0.35], 'alpha': [0.0087, 0.0014]},
		'IALearner.ACFTable': {'beta': [6.3, 0.86], 'alpha': [0.0065, 0.00068]},
		'IAAluisiLearner.ACFTable': {'beta': [6.0, 0.8], 'alpha': [0.0036, 0.00068]},
		'MALearner.ACFTable': {'beta': [5.9, 0.97], 'alpha': [0.0094, 0.0016],
							   'alpha_phi': [0.051, 0.018]},
		'ABIALearner.MFTable': {'nmr': [0.79, 0.067], 'beta': [4.5, 0.29], 'alpha': [0.016, 0.0003],
								'alpha_bias': [0.018, 0.0046]},

		'UABIALearner.FTable': {'nmr': [0.093, 0.21], 'beta': [2.9, 0.54],
								'alpha': [0.024, 0.011],
								'alpha_bias': [0.081, 0.025]},
		'ABIALearner.FTable': {'nmr': [0.72, 0.065], 'beta': [4.3, 0.24],
							   'alpha': [0.0024, 0.00073],
							   'alpha_bias': [0.031, 0.0078]},
		'IALearner.FTable': {'nmr': [0.035, 0.24], 'beta': [4.3, 0.9],
							 'alpha': [0.03, 0.015],
							 'alpha_bias': [0.15, 0.00]},
		'QLearner.QTable': {
			'nmr': [0.035, 0.24], 'beta': [3.6, 0.38],
			'alpha': [0.034, 0.0066], 'alpha_bias': [0.15, 0.00]},
		'UABQLearner.QTable': {'nmr': [0.64, 0.15], 'beta': [2.7, 0.37],
							   'alpha': [0.083, 0.035], 'alpha_bias': [0.0033, 0.0014]},
		'ABQLearner.QTable': {'nmr': [0.035, 0.24], 'beta': [1.9, 0.32],
							  'alpha': [0.084, 0.025], 'alpha_bias': [0.019, 0.0093]},
		'UABQLearner.OptionsTable': {
			'nmr': [0.55, 0.12], 'beta': [2.4, 0.38],
			'alpha': [0.079, 0.029], 'alpha_bias': [0.013, 0.0064]},
		'QLearner.OptionsTable': {'nmr': [0.035, 0.24], 'beta': [3.5, 0.47],
								  'alpha': [0.042, 0.019], 'alpha_bias': [0.15, 0.00]},
		'ABQLearner.OptionsTable': {'nmr': [0.035, 0.24], 'beta': [1.7, 0.19],
									'alpha': [0.073, 0.018], 'alpha_bias': [0.029, 0.013]},


		'QLearner.MQTable': {'nmr': [0.067, 0.24], 'beta': [4.6 ,0.21],
							 'alpha': [0.032, 0.0063],
							 'alpha_bias': [np.nan, np.nan]},

		'QLearner.MOptionsTable': {'nmr': [0.046 , 0.23], 'beta': [4.2,0.35],
								   'alpha': [0.019, 0.0027],
								   'alpha_bias': [np.nan, np.nan]},

		'IALearner.MFTable': {'nmr': [-0.075 , 0.22], 'beta': [3.9, 0.44],
							  'alpha': [0.0072,0.0033],
							  'alpha_bias': [np.nan, np.nan]}
	}

	parameters = estimated_params_values[model_name]
	sampled_simulation_params = [np.random.normal(loc=mean, scale=sem * np.sqrt(10)) for mean, sem in
								 parameters.values()]
	return tuple(np.clip(sampled_simulation_params[i], *ranges[i]) for i, parameter_value in
				 enumerate(sampled_simulation_params))



def run_single_simulation(env, architecture, estimated_parameters, initial_motivation=RewardType.WATER):
	(brain, learner, model) = architecture
	model_instance = model(env.stimuli_encoding_size(), 2, env.num_actions())

	if issubclass(learner, MALearner) or issubclass(learner, DQNAtt):
		(beta, lr, attention_lr) = estimated_parameters
		learner_instance = learner(model_instance, learning_rate=lr, alpha_phi=attention_lr)
	else:
		(beta, lr) = estimated_parameters
		learner_instance = learner(model_instance, learning_rate=lr)

	agent = MotivatedAgent(brain(learner_instance, beta=beta),
						   motivation=initial_motivation, motivated_reward_value=config.MOTIVATED_REWARD,
						   non_motivated_reward_value=0)
	experiment_stats, experiment_data = PlusMazeExperiment(env, agent, dashboard=False)
	return experiment_stats, experiment_data


def run_simulation(env, brains, repetitions=10, initial_motivation=RewardType.WATER):
	all_experiment_data = pd.DataFrame()
	brains_reports = []
	for agent_spec in brains:
		completed_experiments = 0
		aborted_experiments = 0
		brain_repetition_reports = [None] * repetitions
		while completed_experiments < repetitions:
			env.init()
			architecture, parameters_mean, parameters_std, ranges = agent_spec
			#estimated_parameters = parameters

			sampled_simulation_params = [np.random.normal(loc=mean, scale=std) for mean, std in zip(parameters_mean, parameters_std)]

			estimated_parameters = tuple(np.clip(sampled_simulation_params[i], * ranges[i]) for i, parameter_value in
			 			 enumerate(sampled_simulation_params))


			#estimated_parameters = sample_from_estimated_parameters(utils.brain_name(architecture))
			print(estimated_parameters)
			experiment_stats, experiment_data = run_single_simulation(env, architecture, estimated_parameters, initial_motivation)
			experiment_data['initial_motivation'] = initial_motivation.value
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
		'/Users/georgekour/repositories/plus-maze-simulator/fitting/Results/simulations_results/simulation_{}_{}.csv'.format(
			repetitions, utils.get_timestamp()), index=False)

	plot_days_per_stage(brains_reports)

	for brain_report in brains_reports:
		plot_behavior_results(brain_report)



def WPI_simulations():
	models=	[((TDBrain, IALearner, FTable), (0, 2.5, 0.01, 0.1)),#FRL
				((TDBrain, IALearner, MFTable), (0, 2.5, 0.01, 0.1)), #S(V)-FRL
				((TDBrain, UABIALearner, FTable), (0, 2.5, 0.01, 0.1)),#B-FRL
				((TDBrain, ABIALearner, FTable), (0, 2.5, 0.01, 0.1)), # M(B)-FRL
	]

	# models=[	((TDBrain, QLearner, QTable), (0, 2.5, 0.01, 0.1)), #SARL
	# 			((TDBrain, UABQLearner, QTable), (0, 2.5, 0.01, 0.1)), #B-SARL
	# 			((TDBrain, ABQLearner, QTable), (0, 2.5, 0.01, 0.1)), # M(B)-SARL
	# 			]
	#
	# models=[	#((TDBrain, QLearner, OptionsTable), (0, 2.5, 0.01, 0.1)), #SARL
	#
	# 			((TDBrain, UABQLearner, OptionsTable), (0, 2.5, 0.01, 0.1)), #B-SARL
	# 			#((TDBrain, ABQLearner, OptionsTable), (0, 2.5, 0.01, 0.1)), # M(B)-SARL
	# 			]
	run_simulation(PlusMazeOneHotCues(relevant_cue=CueType.ODOR, stimuli_encoding=10), models, repetitions=20)
	#goal_choice_index_model(data_file_path)

	#calcWPI
	""""To capture the goal-choice element, we quantified for each animal in each training session its bias towards one arm pair according to the goal choice index (GC , Fig. 1e), describing the excess of visits to the arms that contained the deprived reward (regardless of whether they made a correct or erroneous choice): GC=\frac{m-um}{m+um} , where m denotes the number of trials that the animal chose an arms in which correct performance would be rewarded deprived reward (i.e., water for water restriction and food for food restriction condition), and um denotes the number of trials where the animal chose one of the two arms which contained the reward that was not restricted (Fig. 1e).  """




def run_motivation_simulations_from_fitting_data(fitting_data_df_file, num_repetitions):
	""" Goven fitting data, extract all models and there parameters, then run simulations while
	tyaking into consideration the mean and std of the estimated parameters."""
	fitting_data_df = pd.read_csv(fitting_data_df_file)

	fitting_data_df['architecture'] = fitting_data_df['model']

	fitting_data_df = fitting_data_df.groupby(['architecture','subject'])['parameters'].first().reset_index()
	# Apply the function to create new columns for learner and model names
	fitting_data_df[['learner', 'model']] = fitting_data_df['architecture'].apply(extract_names_from_architecture).apply(pd.Series)

	models = []
	for archs in fitting_data_df['architecture'].unique():
		df = fitting_data_df[fitting_data_df['architecture']==archs]
		df['parameters'] = df['parameters'].apply(string2list)

		# Calculate the average of each parameter across all subjects
		average_parameters = df['parameters'].apply(pd.Series).mean().tolist()
		std_parameters = df['parameters'].apply(pd.Series).std().tolist()

		learner_class = globals()[df['learner'].iloc[0]]
		model_class = globals()[df['model'].iloc[0]]

		models+=[((TDBrain, learner_class, model_class), tuple(average_parameters), tuple(std_parameters), ([0.1,10], [0.001,0.4], [0.001,0.4]))]

	env = PlusMazeOneHotCues2ActiveDoors(relevant_cue=CueType.ODOR, stimuli_encoding=8)
	run_simulation(env, models, repetitions=num_repetitions, initial_motivation=RewardType.NONE)

if __name__ == '__main__':
	#env = PlusMazeOneHotCues2ActiveDoors(relevant_cue=CueType.ODOR, stimuli_encoding=8)
	# env = PlusMazeOneHotCues(relevant_cue=CueType.ODOR, stimuli_encoding=10)
	# run_simulation(env)
	# x = 1
	#WPI_simulations()
	fitting_file_name = '/Users/georgekour/repositories/plus-maze-simulator/fitting/Results/Rats-Results/fitting_results_AARL_best.csv'
	run_motivation_simulations_from_fitting_data(fitting_file_name, num_repetitions=10)
