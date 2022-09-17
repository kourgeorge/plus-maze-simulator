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
		#axis.set_title("Subject {}".format(i))
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

	fig = plt.figure(figsize=(35, 7), dpi=120, facecolor='w')
	for i, subject in enumerate(np.unique(df["subject"])):

		df_sub = df[df["subject"] == subject]
		axis = fig.add_subplot(330 + i + 1)

		for model in np.unique(df_sub["model"]):
			# find individual best fitting parameters.
			df_sub_model = df_sub[df_sub["model"] == model]
			daily_likl = fitting_utils.float_string_list_to_list(list(df_sub_model["daily_likl"])[0])
			stages = fitting_utils.float_string_list_to_list(list(df_sub_model["stages"])[0])
			days = np.array(range(len(daily_likl)))+1
			axis.plot(days, np.exp(-np.array(daily_likl)), label=model.split('.')[-1])
			axis.xaxis.set_major_locator(MaxNLocator(integer=True))

		axis.set_xlabel('Days') if i>5 else 0
		axis.set_ylabel("Likelihood") if i%3==0 else 0
		axis.set_title("Subject {}".format(i))
		stage_transition_days = np.where(np.ediff1d(stages) == 1)
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

def show_neural_tabular_for_stage_and_subject(data_file_path):
	df = pd.read_csv(data_file_path)
	df.stages_likl = [fitting_utils.float_string_list_to_list(row) for row in list(df.stages_likl)]
	stages = ['ODOR1', 'ODOR2', 'EDShift(Light)']
	for stage_indx, stage in enumerate(stages):
		df[stage] = np.exp(-np.stack(list(df.stages_likl))[:,stage_indx])
	del df['stages_likl']

	df.exp_likl=np.exp(-df.exp_likl)

	subjects = np.unique(df["subject"])

	model_pairs=[('TDBrain.TD.TabularQ','ConsolidationBrain.DQN.FullyConnectedNetwork'),
					  ('TDBrain.TDUniformAttention.UniformAttentionTabular','ConsolidationBrain.DQN.UniformAttentionNetwork'),
					  ('TDBrain.TDUniformAttention.AttentionAtChoiceAndLearningTabular','ConsolidationBrain.DQN.AttentionAtChoiceAndLearningNetwork')]

	results_df = pd.DataFrame()
	for pair_ind, model_pair in enumerate(model_pairs):
		tabular_model = model_pair[0]
		neural_model = model_pair[1]

		for subject in subjects:
			df_sub_model_tabular = df[(df.subject == subject) & (df.model == tabular_model)]
			df_sub_model_neural = df[(df.subject == subject) & (df.model == neural_model)]

			dict={'subject':subject,
				  'pair':model_pair,
				  'expr_likl': (df_sub_model_tabular.exp_likl.iloc[0], df_sub_model_neural.exp_likl.iloc[0]),
				  stages[0]: (df_sub_model_tabular[stages[0]].iloc[0], df_sub_model_neural[stages[0]].iloc[0]),
				  stages[1]: (df_sub_model_tabular[stages[1]].iloc[0], df_sub_model_neural[stages[1]].iloc[0]),
				  stages[2]: (df_sub_model_tabular[stages[2]].iloc[0], df_sub_model_neural[stages[2]].iloc[0])}

			results_df = results_df.append(dict, ignore_index=True)

	sns.set_theme(style="whitegrid")

	fig = plt.figure(figsize=(12, 4), dpi=120, facecolor='w')

	for i, stage in enumerate(stages+['expr_likl']):
		axis = fig.add_subplot(140 + i + 1)
		results_df[stage+'_tabular'] = np.stack(results_df[stage].to_numpy())[:, 0]
		results_df[stage+'_neural'] = np.stack(results_df[stage].to_numpy())[:, 1]

		min_prob= np.min(list(results_df[stage+'_tabular'])+list(results_df[stage+'_neural']))
		max_prob = np.max(list(results_df[stage + '_tabular']) + list(results_df[stage + '_neural']))
		sns.scatterplot(x=stage+'_tabular', y=stage+'_neural', hue='pair', data=results_df, ax=axis)
		#axis.plot(np.linspace(min_prob, max_prob, 100), np.linspace(min_prob, max_prob, 100), color='grey')
		axis.plot(np.linspace(0.1, 0.7, 100), np.linspace(0.1, 0.7, 100), color='grey')
		axis.set(xlabel='Tabular', ylabel='Neural') if i==0 else axis.set(xlabel='Tabular', ylabel='')
		axis.set_title(stage)
		axis.legend([], [], frameon=False)


		handles, labels = axis.get_legend_handles_labels()
		fig.legend(handles, ["Q vs. FC","UA Tabular vs. UA Neural", "ACL Tabular vs. ACL Neural"],
				   loc='upper center', prop={'size': 8})


	plt.subplots_adjust(left=0.1,
						bottom=0.2,
						right=0.9,
						top=0.8,
						wspace=0.6,
						hspace=0.5)
	plt.savefig('fitting/Results/figures/neural_tabular_compare_{}'.format(fitting_utils.get_timestamp()))
	plt.show()


if __name__ == '__main__':
	# show_models_fitting_accuracy_over_times_for_subject(
	# 	PlusMazeOneHotCues2ActiveDoors(relevant_cue=CueType.ODOR, stimuli_encoding=10), maze_models)

	show_neural_tabular_for_stage_and_subject('/Users/gkour/repositories/plusmaze/fitting/Results/Rats-Results/fitting_results_2022_09_17_04_47.csv')
	#show_models_fitting_accuracy_over_times_for_subject_from_data(
	#	'/Users/gkour/repositories/plusmaze/fitting/Results/Rats-Results/fitting_results_2022_09_17_04_47.csv')

	x = 1
