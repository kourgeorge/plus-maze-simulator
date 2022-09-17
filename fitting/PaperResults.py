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

stages = ['ODOR1', 'ODOR2', 'EDShift(Light)']


def models_fitting_quality_over_times(data_file_path):
	df = pd.read_csv(data_file_path)
	df = df[['subject', 'model', 'stage', 'day in stage', 'trial', 'likelihood']].copy()

	fig = plt.figure(figsize=(35, 7), dpi=120, facecolor='w')
	for i, subject in enumerate(np.unique(df["subject"])):

		df_sub = df[df["subject"] == subject]
		axis = fig.add_subplot(330 + i + 1)

		for model in np.unique(df_sub["model"]):
			df_sub_model = df_sub[df_sub["model"] == model]

			model_subject_df = df_sub_model.groupby(['subject', 'model', 'stage', 'day in stage']).mean().reset_index()
			model_subject_df.likelihood = np.exp(-model_subject_df.likelihood)

			days = list(model_subject_df.index + 1)
			axis.plot(days, model_subject_df.likelihood, label=model.split('.')[-1])
			axis.xaxis.set_major_locator(MaxNLocator(integer=True))

		axis.set_xlabel('Days') if i > 5 else 0
		axis.set_ylabel("Likelihood") if i % 3 == 0 else 0
		axis.set_title("Subject {}".format(i))
		stage_transition_days = np.where(model_subject_df['day in stage'] == 1)[0][1:]
		for stage_day in stage_transition_days:
			axis.axvline(x=stage_day + 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2)

		axis.set_ylim(0.1, 0.7)

	handles, labels = axis.get_legend_handles_labels()
	fig.legend(handles, labels, prop={'size': 8.5})  # loc=(0.55,0.1), prop={'size': 7}

	plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

	plt.savefig('fitting/Results/figures/Likelihood_{}'.format(fitting_utils.get_timestamp()))
	plt.show()


def compare_neural_tabular_models(data_file_path):
	df = pd.read_csv(data_file_path)
	df = df[['subject', 'model', 'stage', 'day in stage', 'trial', 'likelihood']].copy()

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
		sns.scatterplot(x='likelihood_x', y='likelihood_y', hue='pair', size='day in stage', data=pairs_df_stage,
						ax=axis, alpha=0.6, s=20)
		axis.plot(np.linspace(0.1, 0.7, 100), np.linspace(0.1, 0.7, 100), color='grey')
		axis.set(xlabel='Tabular', ylabel='Neural') if i == 0 else axis.set(xlabel='Tabular', ylabel='')
		axis.set_title(stage)
		axis.legend([], [], frameon=False)

	axis = fig.add_subplot(144)
	sns.scatterplot(x='likelihood_x', y='likelihood_y', hue='pair', data=pairs_df, ax=axis, alpha=0.5, s=10)
	axis.plot(np.linspace(0.1, 0.7, 100), np.linspace(0.1, 0.7, 100), color='grey')
	axis.set_title('All Stages')
	axis.set(xlabel='Tabular', ylabel='')
	axis.legend([], [], frameon=False)

	handles, labels = axis.get_legend_handles_labels()
	fig.legend(handles, ["Q vs. FC", "UA Tabular vs. UA Neural", "ACL Tabular vs. ACL Neural"],
			   loc='upper center', prop={'size': 8})

	plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.8, wspace=0.6, hspace=0.5)
	plt.savefig('fitting/Results/figures/neural_tabular_compare_{}'.format(fitting_utils.get_timestamp()))
	plt.show()


def compare_model_subject_learning_curve(data_file_path):
	df = pd.read_csv(data_file_path)
	df = df[['subject', 'model', 'stage', 'day in stage', 'trial', 'reward', 'model_reward']].copy()

	days_info_df = df.groupby(['subject', 'model', 'stage', 'day in stage']).mean().reset_index()

	fig = plt.figure(figsize=(35, 7), dpi=120, facecolor='w')
	for i, subject in enumerate(np.unique(df["subject"])):
		df_sub = days_info_df[days_info_df["subject"] == subject]
		axis = fig.add_subplot(330 + i + 1)

		for model in np.unique(df_sub["model"]):
			model_subject_df = df_sub[df_sub["model"] == model]
			days = range(len(model_subject_df))
			axis.plot(days, model_subject_df.model_reward, label=model, alpha=0.5)
			axis.xaxis.set_major_locator(MaxNLocator(integer=True))

			axis.set_xlabel('Days') if i > 5 else 0
			axis.set_ylabel("Accuracy") if i % 3 == 0 else 0

		stage_transition_days = np.where(model_subject_df['day in stage'] == 1)[0][1:]
		for stage_day in stage_transition_days:
			axis.axvline(x=stage_day + 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2)

		axis.plot(days, model_subject_df.reward, label='subject', color='black')

	handles, labels = axis.get_legend_handles_labels()
	fig.legend(handles, labels, prop={'size': 8.5})  # loc=(0.55,0.1), prop={'size': 7}

	plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

	plt.savefig('fitting/Results/figures/learning_curve_{}'.format(fitting_utils.get_timestamp()))
	plt.show()


def plot_models_fitting_result_per_stage(data_file_path):
	df = pd.read_csv(data_file_path)

	df = df[['subject', 'model', 'stage', 'day in stage', 'trial', 'likelihood']].copy()
	df.likelihood = np.exp(-df.likelihood)
	df_stage = df.groupby(['subject', 'model', 'stage'], sort=False).mean().reset_index()
	sns.set_theme(style="whitegrid")
	# _, idx = np.unique(df_stage.model, return_index=True)
	# order = (df_stage.model[np.sort(idx)])

	fig = plt.figure(figsize=(11, 5), layout="constrained")
	spec = fig.add_gridspec(1, 3)
	ax0 = fig.add_subplot(spec[0, 0:2])
	ax1 = fig.add_subplot(spec[0, 2])

	g1 = sns.boxplot(x='stage', y='likelihood', hue='model', data=df_stage, ax=ax0)
	g1.set_xticklabels(stages)
	g1.set(xlabel='', ylabel='Likelihood')
	g1.legend([], [], frameon=False)
	df['dummy']=1
	df = df.groupby(['subject', 'model'], sort=False).mean().reset_index()
	g2 = sns.boxplot(x='dummy', y='likelihood', hue='model', data=df, ax=ax1)
	g2.set_xticklabels([''])
	g2.set(xlabel='All Stages', ylabel='')
	g2.set_ylim(g1.get_ylim())
	g2.legend([], [], frameon=False)
	plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.8, wspace=0.3, hspace=0.3)

	handles, labels = g2.get_legend_handles_labels()
	fig.legend(handles, labels, loc='upper right', prop={'size': 9.5})

	plt.savefig('fitting/Results/figures/all_models_by_stage_{}'.format(fitting_utils.get_timestamp()))
	plt.show()


# # plots.plot_histogram(result=brain_results, dimension_identifier='lr', bins=20)
# # plots.plot_objective_2D(brain_results['results'], 'lr', 'batch_size')
# # plots.plot_objective(brain_results['results'], plot_dims=['nmr', 'lr'])


if __name__ == '__main__':
	file_path = '/Users/gkour/repositories/plusmaze/fitting/Results/Rats-Results/fitting_results_2022_09_17_21_03.csv'
	# models_fitting_quality_over_times(file_path)
	# compare_neural_tabular_models(file_path)
	# compare_model_subject_learning_curve(file_path)
	plot_models_fitting_result_per_stage(file_path)

	x = 1
