import numpy as np

from environment import PlusMazeOneHotCues2ActiveDoors, CueType
import pandas as pd
import os
import matplotlib.pyplot as plt

from fitting import fitting_utils
from fitting.MazeBayesianModelFitting import MazeBayesianModelFitting
from fitting.fitting_config import MAZE_ANIMAL_DATA_PATH, maze_models
from matplotlib.ticker import MaxNLocator
import seaborn as sns


def show_models_fitting_accuracy_over_times_for_subject(env, models):
	# show likelihood per day, indicate the stages, etc.
	all_subjects_data = [pd.read_csv(os.path.join(MAZE_ANIMAL_DATA_PATH, rat_file))
						 for rat_file in list(np.sort(os.listdir(MAZE_ANIMAL_DATA_PATH)))]
	fig = plt.figure(figsize=(15, 15), dpi=120, facecolor='w')
	for i, experiment_data in enumerate(all_subjects_data):
		axis = fig.add_subplot(520 + i + 1)

		for model in models:
			# find individual best fitting parameters.

			model, parameters_space = model
			fitter = MazeBayesianModelFitting(env, experiment_data, model, parameters_space, n_calls=11)
			search_result, experiment_stats, all_experiment_likelihoods = fitter._fake_optimize()
			axis.plot(np.exp(-experiment_stats.epoch_stats_df.Likelihood.to_numpy()),
					  label=fitting_utils.brain_name(model))

		axis.xlabel('Days')
		axis.set_ylabel("Subject {}".format(i))
		# axis.set_title("Subject {}".format(i))
		stage_transition_days = np.where(np.ediff1d(experiment_stats.epoch_stats_df.Stage) == 1)
		for stage_day in stage_transition_days[0] + 1:
			axis.axvline(x=stage_day + 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2)

	handles, labels = axis.get_legend_handles_labels()
	fig.legend(handles, labels, loc=(0.55, 0.1))  # loc='lower right', prop={'size': 7}

	plt.subplots_adjust(left=0.1,
						bottom=0.1,
						right=0.9,
						top=0.9,
						wspace=0.4,
						hspace=0.4)

	# fig.legend(prop={'size': 7})

	plt.savefig('fitting/Results/figures/Likelihood_{}'.format(fitting_utils.get_timestamp()))
	plt.show()


def show_models_fitting_accuracy_over_times_for_subject_from_data(data_file_path):
	df = pd.read_csv(data_file_path)
	df = df[['subject', 'model', 'stage', 'day in stage', 'trial', 'likelihood']].copy()

	fig = plt.figure(figsize=(35, 7), dpi=120, facecolor='w')
	for i, subject in enumerate(np.unique(df["subject"])):

		df_sub = df[df["subject"] == subject]
		axis = fig.add_subplot(330 + i + 1)

		for model in np.unique(df_sub["model"]):
			# find individual best fitting parameters.
			df_sub_model = df_sub[df_sub["model"] == model]
			#daily_likl = fitting_utils.float_string_list_to_list(list(df_sub_model["daily_likl"])[0])

			model_subject_df = df_sub_model.groupby(['subject', 'model', 'stage', 'day in stage']).mean().reset_index()
			model_subject_df.likelihood = np.exp(-model_subject_df.likelihood)
			daily_likl = model_subject_df.likelihood

			#stages = fitting_utils.float_string_list_to_list(list(df_sub_model["stages"])[0])
			days = list(model_subject_df.index+1)
			axis.plot(days, daily_likl, label=model.split('.')[-1])
			axis.xaxis.set_major_locator(MaxNLocator(integer=True))

		axis.set_xlabel('Days') if i > 5 else 0
		axis.set_ylabel("Likelihood") if i % 3 == 0 else 0
		axis.set_title("Subject {}".format(i))
		stage_transition_days = np.where(model_subject_df['day in stage'] == 1)
		for stage_day in stage_transition_days[0] + 1:
			axis.axvline(x=stage_day + 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2)

		axis.set_ylim(0.1, 0.7)

	handles, labels = axis.get_legend_handles_labels()
	fig.legend(handles, labels, prop={'size': 8.5})  # loc=(0.55,0.1), prop={'size': 7}

	plt.subplots_adjust(left=0.1,
						bottom=0.1,
						right=0.9,
						top=0.9,
						wspace=0.4,
						hspace=0.4)

	# fig.legend(prop={'size': 7})

	plt.savefig('fitting/Results/figures/Likelihood_{}'.format(fitting_utils.get_timestamp()))
	plt.show()


def compare_neural_tabular_models(data_file_path):
	df = pd.read_csv(data_file_path)
	df = df[['subject', 'model', 'stage', 'day in stage', 'trial', 'likelihood']].copy()
	stages = ['ODOR1', 'ODOR2', 'EDShift(Light)']

	stage_mean_df = df.groupby(['subject', 'model', 'stage', 'day in stage']).mean().reset_index()
	stage_mean_df.likelihood = np.exp(-stage_mean_df.likelihood)

	model_pairs = [('TabularQ', 'FullyConnectedNetwork'),
				   ('UniformAttentionTabular', 'UniformAttentionNetwork'),
				   ('AttentionAtChoiceAndLearningTabular', 'AttentionAtChoiceAndLearningNetwork')]
	pairs_df = pd.DataFrame()
	joined_df = stage_mean_df.merge(stage_mean_df, on=['subject', 'stage', 'day in stage'])

	for pair_ind, model_pair in enumerate(model_pairs):
		tabular_model = model_pair[0]
		neural_model = model_pair[1]
		pair_df = joined_df[(joined_df.model_x == tabular_model) & (joined_df.model_y == neural_model)]
		pairs_df = pairs_df.append(pair_df, ignore_index=True)
	pairs_df['pair'] = pairs_df.model_x + ', ' + pairs_df.model_y

	sns.set_theme(style="whitegrid")
	fig = plt.figure(figsize=(12, 4), dpi=120, facecolor='w')
	for i, stage in enumerate(stages):
		axis = fig.add_subplot(140 + i + 1)
		pairs_df_stage = pairs_df[pairs_df['stage'] == i + 1]
		sns.scatterplot(x='likelihood_x', y='likelihood_y', hue='pair', data=pairs_df_stage, ax=axis, alpha=0.6, s=20)
		axis.plot(np.linspace(0.1, 0.7, 100), np.linspace(0.1, 0.7, 100), color='grey')
		axis.set(xlabel='Tabular', ylabel='Neural') if i == 0 else axis.set(xlabel='Tabular', ylabel='')
		axis.set_title(stage)
		axis.legend([], [], frameon=False)

		handles, labels = axis.get_legend_handles_labels()
		fig.legend(handles, ["Q vs. FC", "UA Tabular vs. UA Neural", "ACL Tabular vs. ACL Neural"],
				   loc='upper center', prop={'size': 8})

	axis = fig.add_subplot(144)
	sns.scatterplot(x='likelihood_x', y='likelihood_y', hue='pair', data=pairs_df, ax=axis, alpha=0.5, s=10)
	axis.plot(np.linspace(0.1, 0.7, 100), np.linspace(0.1, 0.7, 100), color='grey')
	axis.set_title('All Stages')
	axis.set(xlabel='Tabular', ylabel='')
	axis.legend([], [], frameon=False)


	plt.subplots_adjust(left=0.1,
						bottom=0.2,
						right=0.9,
						top=0.8,
						wspace=0.6,
						hspace=0.5)
	plt.savefig('fitting/Results/figures/neural_tabular_compare_{}'.format(fitting_utils.get_timestamp()))
	plt.show()


if __name__ == '__main__':


	show_models_fitting_accuracy_over_times_for_subject_from_data(
		'/Users/gkour/repositories/plusmaze/fitting/Results/Rats-Results/fitting_results_2022_09_17_17_15.csv')
	# compare_neural_tabular_models(
	# 	'/Users/gkour/repositories/plusmaze/fitting/Results/Rats-Results/fitting_results_2022_09_17_17_15.csv')
	x = 1
