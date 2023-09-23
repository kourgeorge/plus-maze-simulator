__author__ = 'gkour'

import pandas as pd

from PlusMazeExperiment import PlusMazeExperiment, ExperimentStatus
from behavioral_analysis import plot_days_per_stage, plot_behavior_results
from brains.consolidationbrain import ConsolidationBrain
from brains.tdbrain import TDBrain
from environment import CueType, PlusMazeOneHotCues, PlusMazeOneHotCues2ActiveDoors, StagesTransition, PlusMaze
from fitting.fitting_utils import extract_names_from_architecture, string2list
from fitting.simulation_utils import sample_brain_parameters, extract_model_average_fitting_parameters
from learners.networklearners import *
from learners.tabularlearners import *
from models.networkmodels import *
from models.tabularmodels import *
from motivatedagent import MotivatedAgent
from rewardtype import RewardType


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


def run_simulation_sampled_brain(env: PlusMaze, brains_parameters_ranges, repetitions=10,
								 initial_motivation: RewardType = RewardType.WATER):
	"""Given a PlusMaze environment and a set of agent architectures, runs a simulation of the agent on the environment."""

	all_simulation_data = pd.DataFrame()
	brains_reports = []
	for agent_spec in brains_parameters_ranges:
		completed_experiments = 0
		aborted_experiments = 0
		brain_repetition_reports = [None] * repetitions
		while completed_experiments < repetitions:
			env.init()
			architecture, parameters_mean, parameters_std, ranges = agent_spec
			#estimated_parameters = parameters

			estimated_parameters = sample_brain_parameters(parameters_mean, parameters_std, ranges)

			#estimated_parameters = sample_from_estimated_parameters(utils.brain_name(architecture))
			print(estimated_parameters)
			experiment_stats, experiment_data = run_single_simulation(env, architecture, estimated_parameters, initial_motivation)
			experiment_data['initial_motivation'] = initial_motivation.value
			experiment_data['model'] = utils.brain_name(architecture)
			experiment_data['subject'] = -1*completed_experiments
			experiment_data['parameters'] = [estimated_parameters]*len(experiment_data)

			if experiment_stats.metadata['experiment_status'] == ExperimentStatus.COMPLETED:
				brain_repetition_reports[completed_experiments] = experiment_stats
				completed_experiments += 1
				all_simulation_data = pd.concat([all_simulation_data,experiment_data])
			else:
				aborted_experiments += 1
		brains_reports.append(brain_repetition_reports)
		print("{} out of {} experiments were aborted".format(aborted_experiments,
															 aborted_experiments + completed_experiments))


	return all_simulation_data
	# all_experiment_data.to_csv(
	# 	'/Users/georgekour/repositories/plus-maze-simulator/fitting/Results/simulations_results/simulation_{}_{}.csv'.format(
	# 		repetitions, utils.get_timestamp()), index=False)

	# plot_days_per_stage(brains_reports)
	#
	# for brain_report in brains_reports:
	# 	plot_behavior_results(brain_report)



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
	run_simulation_sampled_brain(PlusMazeOneHotCues(relevant_cue=CueType.ODOR, stimuli_encoding=10), models, repetitions=20)
	#goal_choice_index_model(data_file_path)

	#calcWPI
	""""To capture the goal-choice element, we quantified for each animal in each training session its bias towards one arm pair according to the goal choice index (GC , Fig. 1e), describing the excess of visits to the arms that contained the deprived reward (regardless of whether they made a correct or erroneous choice): GC=\frac{m-um}{m+um} , where m denotes the number of trials that the animal chose an arms in which correct performance would be rewarded deprived reward (i.e., water for water restriction and food for food restriction condition), and um denotes the number of trials where the animal chose one of the two arms which contained the reward that was not restricted (Fig. 1e).  """



def run_motivation_simulations_from_fitting_data(fitting_data_df_file, num_repetitions, stages=PlusMazeOneHotCues2ActiveDoors.default_stages):
	""" Given fitting data, extract all models and there parameters, then run simulations while
	taking into consideration the mean and std of the estimated parameters."""
	fitting_data_df = pd.read_csv(fitting_data_df_file)

	fitting_data_df['architecture'] = fitting_data_df['model']

	fitting_data_df = fitting_data_df.groupby(['architecture','subject'])['parameters'].first().reset_index()
	# Apply the function to create new columns for learner and model names

	models = []
	for arch in fitting_data_df['model'].unique():
		df = fitting_data_df[fitting_data_df['model']==arch]
		df['parameters'] = df['parameters'].apply(string2list)

		# Calculate the average of each parameter across all subjects
		average_parameters = df['parameters'].apply(pd.Series).mean().tolist()
		std_parameters = df['parameters'].apply(pd.Series).std().tolist()

		leaner_name, model_name = extract_names_from_architecture(arch)
		learner_class = globals()[leaner_name]
		model_class = globals()[model_name]

		models+=[((TDBrain, learner_class, model_class), tuple(average_parameters), tuple(std_parameters), ([0.1,10], [0.001,0.4], [0.001,0.4]))]

	env = PlusMazeOneHotCues2ActiveDoors(stages=stages, relevant_cue=CueType.ODOR, stimuli_encoding=8)
	run_simulation_sampled_brain(env, models, repetitions=num_repetitions, initial_motivation=RewardType.NONE)


def run_increasing_IDShift(fitting_data_df_file, repetitions=50):
	average_parameters, std_parameters = extract_model_average_fitting_parameters(fitting_data_df_file,
																				  'MALearner.ACFTable')
	stages = [{'name': 'LED', 'transition_logic': StagesTransition.set_color_stage}]
	all_simulation_data = pd.DataFrame()
	for i in range(6):
		env = PlusMazeOneHotCues2ActiveDoors(stages=stages, relevant_cue=CueType.ODOR, stimuli_encoding=14)

		env_df = run_simulation_sampled_brain(env=env, brains_parameters_ranges=[((TDBrain, MALearner, ACFTable),
																		average_parameters, std_parameters,
																		 ([0.1, 10], [0.001, 0.4], [0.001, 0.4]))],
									 repetitions=repetitions, initial_motivation=RewardType.NONE)

		env_df["env_setup"] = ",".join([stage['name'] for stage in stages])
		all_simulation_data = pd.concat([all_simulation_data, env_df])
		stages.insert(0, {'name': f'Odor{i+1}', 'transition_logic': StagesTransition.set_odor_stage})

	return all_simulation_data




if __name__ == '__main__':
	fitting_file_name = '/Users/georgekour/repositories/plus-maze-simulator/fitting/Results/Rats-Results/reported_results_dimensional_shifting/main_results_reported_10_1_recalculated.csv'
	all_simulation_data = run_increasing_IDShift(fitting_file_name)
	all_simulation_data.to_csv(path_or_buf="/Users/georgekour/repositories/plus-maze-simulator/fitting/Results/simulations_results/increasing_ID_50.csv", index=False)
	#run_motivation_simulations_from_fitting_data(fitting_file_name, num_repetitions=10)
