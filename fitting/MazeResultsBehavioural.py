import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
import functools

import utils
from fitting import fitting_utils
from fitting.fitting_utils import stable_unique, rename_models, models_order_df
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import json

plt.rcParams.update({'font.size': 16})

stages = ['ODOR1', 'ODOR2', 'LED']
num_days_reported = [4, 2, 7]


def despine(axis):
	axis.spines['top'].set_visible(False)
	axis.spines['right'].set_visible(False)


def models_fitting_quality_over_times_average(data_file_path):
	df = pd.read_csv(data_file_path)
	df = df[['subject', 'model', 'stage', 'day in stage', 'trial', 'likelihood', 'reward', 'model_reward']].copy()
	df['NLL'] = -np.log(df.likelihood)

	model_df = df.groupby(['subject', 'model', 'stage', 'day in stage'], sort=False).mean().reset_index()
	model_df['ML'] = np.exp(-model_df.NLL)

	model_df = rename_models(model_df)

	model_df = model_df[~((model_df.stage == 1) & (model_df['day in stage'] > num_days_reported[0]))]
	model_df = model_df[~((model_df.stage == 2) & (model_df['day in stage'] > num_days_reported[1]))]
	model_df = model_df[~((model_df.stage == 3) & (model_df['day in stage'] > num_days_reported[2]))]

	model_df['ind'] = model_df.stage+0.1*model_df['day in stage']
	model_df['ind'] = model_df['ind'].astype(str)

	fig = plt.figure(figsize=(7.5, 4), dpi=100, facecolor='w')
	axis = sns.lineplot(x="ind", y="likelihood", hue="model", hue_order=models_order_df(model_df),
						data=model_df, errorbar="se", err_style='band')
	for stage_day in [3, 5]:
		axis.axvline(x=stage_day + 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2, color='gray')

	axis.set_xlabel('Stage.Day')
	axis.set_ylabel('Average Likelihood')

	handles, labels = axis.get_legend_handles_labels()
	plt.legend(handles, labels, loc="upper left", prop={'size': 14}, labelspacing=0.2)
	axis.spines['top'].set_visible(False)
	axis.spines['right'].set_visible(False)
	plt.subplots_adjust(left=0.12, bottom=0.15, right=0.98, top=0.98, wspace=0.2, hspace=0.1)

def models_fitting_quality_over_times(data_file_path):
	df = pd.read_csv(data_file_path)
	df = df[['subject', 'model', 'stage', 'day in stage', 'trial', 'likelihood', 'reward', 'model_reward']].copy()
	df['NLL'] = -np.log(df.likelihood)

	fig = plt.figure(figsize=(35, 7), dpi=120, facecolor='w')
	for i, subject in enumerate(stable_unique(df["subject"])):
		df_sub = df[df["subject"] == subject]
		axis = fig.add_subplot(330 + i + 1)

		for model in stable_unique(df_sub["model"]):
			df_sub_model = df_sub[df_sub["model"] == model]

			model_subject_df = df_sub_model.groupby(['subject', 'model', 'stage', 'day in stage'], sort=False).mean().reset_index()
			days = list(model_subject_df.index + 1)
			model_subject_df['ML'] = np.exp(-model_subject_df.NLL)
			axis.plot(days, model_subject_df.ML, label=model, alpha=0.6)
			axis.xaxis.set_major_locator(MaxNLocator(integer=True))
			#axis.set_yticklabels(['']) if i % 3 != 0 else 0

		axis.set_xlabel('Days') if i > 5 else 0
		axis.set_ylabel("Likelihood") if i % 3 == 0 else 0
		axis.set_title("Subject {}".format(i+1))
		stage_transition_days = np.where(model_subject_df['day in stage'] == 1)[0][1:]
		for stage_day in stage_transition_days:
			axis.axvline(x=stage_day + 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2)

		#axis.set_ylim(0.45, 1)
		axis.axhline(y=0.5, alpha=0.7, lw=1, color='grey', linestyle='--')

	handles, labels = axis.get_legend_handles_labels()
	fig.legend(handles, labels, loc="upper left", prop={'size': 9}, labelspacing=0.3)  # loc=(0.55,0.1), prop={'size': 7}

	plt.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=0.8, wspace=0.1, hspace=0.4)

	plt.savefig('fitting/Results/figures/daily_likl_{}'.format(fitting_utils.get_timestamp()))
	plt.show()


def compare_neural_tabular_models(data_file_path):
	df = pd.read_csv(data_file_path)
	df = df[['subject', 'model', 'stage', 'day in stage', 'trial', 'likelihood']].copy()
	df.dropna(inplace=True)
	df = rename_models(df)

	stage_mean_df = df.groupby(['subject', 'model', 'stage', 'day in stage']).median().reset_index()

	model_pairs = [('TLR', 'NRL'),
				   ('UA', 'MUA'),
				   ('MUA', 'AAM')]
	stage_mean_df = stage_mean_df[stage_mean_df.model.isin(sum(model_pairs,()))]
	pairs_df = pd.DataFrame()
	joined_df = stage_mean_df.merge(stage_mean_df, on=['subject', 'stage', 'day in stage'])

	for pair_ind, model_pair in enumerate(model_pairs):
		tabular_model = model_pair[0]
		neural_model = model_pair[1]
		pair_df = joined_df[(joined_df.model_x == tabular_model) & (joined_df.model_y == neural_model)]
		pairs_df = pairs_df.append(pair_df, ignore_index=True)
	pairs_df['pair'] = pairs_df.model_x + ', ' + pairs_df.model_y

	minn = 0
	maxx = 1
	sns.set_theme(style="whitegrid")
	fig = plt.figure(figsize=(12, 4), dpi=120, facecolor='w')
	for i, stage in enumerate(stages):
		axis = fig.add_subplot(140 + i + 1)
		pairs_df_stage = pairs_df[pairs_df['stage'] == i + 1]
		pairs_df_stage = pairs_df_stage.rename(columns={'likelihood_x': 'Likelihood1', 'likelihood_y': 'Likelihood2'})
		sns.scatterplot(x='Likelihood1', y='Likelihood2', hue='pair', size='day in stage', data=pairs_df_stage,
						ax=axis, alpha=0.6, s=20)
		axis.plot(np.linspace(minn, maxx, 100), np.linspace(minn, maxx, 100), color='grey')
		#axis.set(xlabel='Tabular', ylabel='Neural') if i == 0 else axis.set(xlabel='Tabular', ylabel='')
		axis.set_title(stage)
		axis.legend([], [], frameon=False)
		axis.set_ylim(minn, maxx)
		axis.set_yticklabels(['']) if i > 0 else 0

	axis = fig.add_subplot(144)
	pairs_df = pairs_df.rename(columns={'likelihood_x': 'Likelihood1', 'likelihood_y': 'Likelihood2'})
	sns.scatterplot(x='Likelihood1', y='Likelihood2', hue='pair', data=pairs_df, ax=axis, alpha=0.5, s=10)
	axis.plot(np.linspace(minn, maxx, 100), np.linspace(minn, maxx, 100), color='grey')
	axis.set_title('All Stages')
	#axis.set(xlabel='Tabular', ylabel='')
	axis.legend([], [], frameon=False)
	axis.set_ylim(minn, maxx)
	axis.set_yticklabels([''])

	handles, labels = axis.get_legend_handles_labels()
	fig.legend(handles, labels, loc='upper left', prop={'size': 8.5})

	plt.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.8, wspace=0.2, hspace=0.2)
	plt.savefig('fitting/Results/figures/neural_tabular_compare_{}'.format(fitting_utils.get_timestamp()))
	#plt.show()


def compare_model_subject_learning_curve_average(data_file_path):
	df = pd.read_csv(data_file_path)
	df = df[['subject', 'model', 'stage', 'day in stage', 'trial', 'reward', 'model_reward']].copy()

	model_df = df.groupby(['subject', 'model', 'stage', 'day in stage'], sort=False).mean().reset_index()

	model_df = rename_models(model_df)
	model_df = model_df[~((model_df.stage == 1) & (model_df['day in stage'] > num_days_reported[0]))]
	model_df = model_df[~((model_df.stage == 2) & (model_df['day in stage'] > num_days_reported[1]))]
	model_df = model_df[~((model_df.stage == 3) & (model_df['day in stage'] > num_days_reported[2]))]

	model_df['ind'] = model_df.stage+0.1*model_df['day in stage']
	model_df['ind'] = model_df['ind'].astype(str)

	fig = plt.figure(figsize=(7.5, 4), dpi=100, facecolor='w')
	axis = sns.lineplot(x="ind", y="model_reward", hue="model", hue_order=models_order_df(model_df),
						data=model_df, errorbar="se", err_style='band')

	subject_reward_df = model_df[['ind','subject', 'stage', 'day in stage', 'trial', 'reward']].copy().drop_duplicates()
	axis = sns.lineplot(x="ind", y="reward", data=subject_reward_df, errorbar="se", err_style='band', linestyle='--',
						ax=axis, color='black')

	#plt.legend(handles=plt.legend().legendHandles, labels=['a','b','c','d','e'])
	#axis = sns.barplot(x="ind", y="reward", data=subject_reward_df, errorbar="se")
	for stage_day in [3, 5]:
		axis.axvline(x=stage_day + 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2, color='grey')

	axis.set_xlabel('Stage.Day')
	axis.set_ylabel('Success Rate')

	handles, labels = axis.get_legend_handles_labels()
	plt.legend(handles, labels, loc="upper left", prop={'size': 14}, labelspacing=0.2)
	axis.spines['top'].set_visible(False)
	axis.spines['right'].set_visible(False)
	plt.subplots_adjust(left=0.12, bottom=0.15, right=0.98, top=0.98, wspace=0.2, hspace=0.1)

	plt.savefig('fitting/Results/paper_figures/learning_curve_{}'.format(utils.get_timestamp()))


def learning_curve_behavioral_boxplot(data_file_path):
	plt.rcParams.update({'font.size': 12})
	df = pd.read_csv(data_file_path)
	df = df[['subject', 'model', 'stage', 'day in stage', 'trial', 'reward', 'model_reward']].copy()
	df = df[df.model == df.model[0]]
	days_info_df = df.groupby(['subject', 'model', 'stage', 'day in stage'], sort=False).mean().reset_index()

	days_info_df['ind'] = days_info_df.apply(lambda x:str(x.stage)+str(x['day in stage']), axis='columns')

	def compare(x, y):
		if int(x[0]) < int(y[0]):
			return -1
		elif int(x[0]) > int(y[0]):
			return 1
		elif int(x[1:]) < int(y[1:]):
			return -1
		else:
			return 1

	order = sorted(np.unique(days_info_df['ind']), key=functools.cmp_to_key(compare))

	axis = sns.boxplot(data=days_info_df, x='ind', y='reward',  order=order, palette="flare")

	for stage_day in [8, 11]:
		axis.axvline(x=stage_day + 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2, color='grey')

	animals_in_day = [len(np.unique(days_info_df[days_info_df.ind==day_stage].subject)) for day_stage in order]
	axis.axhline(y=0.5, alpha=0.7, lw=1, color='grey', linestyle='--')

	#add count of animals in each day.
	axis.set_ylabel([0.4,1])
	for xtick in axis.get_xticks():
		axis.text(xtick, 0.41, animals_in_day[xtick],
				horizontalalignment='center',size='small',color='black',weight='semibold')

	ticks = ["{}".format(int(x[1:])) if (int(x[1:])-1)%3==0 else "" for ind, x in enumerate(order) ]
	axis.set_xticklabels(ticks)

	axis.set_xlabel('Training Day in Stage')
	axis.set_ylabel('Success rate')

	axis.spines['top'].set_visible(False)
	axis.spines['right'].set_visible(False)

	plt.subplots_adjust(left=0.08, bottom=0.15, right=0.99, top=0.99, wspace=0.1, hspace=0.4)


def show_days_to_criterion(data_file_path):
	df = pd.read_csv(data_file_path)

	df = df[df.model==df.model[0]]
	df = df[['subject', 'stage', 'day in stage', 'model', 'trial', 'reward']].copy()

	df = rename_models(df)
	df = fitting_utils.cut_off_data_when_reaching_criterion(df)
	sns.set_palette('OrRd', n_colors=len(stages))
	df = df.groupby(['subject', 'model', 'stage'], sort=False).agg({'day in stage': 'max'}).reset_index()
	fig = plt.figure(figsize=(4, 4))
	g1 = sns.barplot(x='stage', y='day in stage', order=list(range(1, len(stages)+1)),
					 data=df, errorbar='se', errwidth=1, capsize=.05)

	g1.set(xlabel='', ylabel='Days Until Criterion')

	pairs = [((2), (3)), ((1), (2)), ((1), (3)), ]
	annot = Annotator(g1, pairs, x='stage', y='day in stage', data=df)
	annot.configure(test='t-test_paired', text_format='star', loc='inside', verbose=2, line_height=0.05,
					comparisons_correction="Bonferroni")
	annot.apply_test().annotate()

	g1.set_xticklabels(stages)
	despine(g1)
	plt.subplots_adjust(left=0.2, bottom=0.1, right=0.99, top=0.95, wspace=0.1, hspace=0.4)

	x=1


def compare_model_subject_learning_curve(data_file_path):
	df = pd.read_csv(data_file_path)
	df = df[['subject', 'model', 'stage', 'day in stage', 'trial', 'reward', 'model_reward']].copy()

	days_info_df = df.groupby(['subject', 'model', 'stage', 'day in stage'], sort=False).mean().reset_index()

	fig = plt.figure(figsize=(35, 7), dpi=120, facecolor='w')
	for i, subject in enumerate(stable_unique(df["subject"])):
		df_sub = days_info_df[days_info_df["subject"] == subject]
		axis = fig.add_subplot(330 + i + 1)

		for model in stable_unique(df_sub["model"]):
			model_subject_df = df_sub[df_sub["model"] == model].reset_index()
			days = list(model_subject_df.index + 1)
			axis.plot(days, model_subject_df.model_reward, label=model, alpha=0.7)
			axis.xaxis.set_major_locator(MaxNLocator(integer=True))

			axis.set_title("Subject {}".format(i + 1))
			axis.set_xlabel('Days') if i > 5 else 0
			axis.set_ylabel("Accuracy") if i % 3 == 0 else 0
			axis.set_yticklabels(['']) if i % 3 != 0 else 0

		stage_transition_days = np.where(model_subject_df['day in stage'] == 1)[0][1:]
		for stage_day in stage_transition_days:
			axis.axvline(x=stage_day + 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2)

		axis.plot(days, model_subject_df.reward, label='subject', color='black')
		axis.axhline(y=0.5, alpha=0.7, lw=1, color='grey', linestyle='--')
		axis.axhline(y=0.75, alpha=0.5, lw=0.7, color='blue', linestyle='--')
		axis.set_ylim([0.3, 1.05])

	handles, labels = axis.get_legend_handles_labels()
	fig.legend(handles, labels, loc=(0.01, 0.82), prop={'size': 8}, labelspacing=0.3)  # loc=(0.55,0.1), prop={'size': 7}

	plt.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=0.8, wspace=0.1, hspace=0.4)

	axis.spines['top'].set_visible(False)
	axis.spines['right'].set_visible(False)

	plt.savefig('fitting/Results/figures/learning_curve_{}'.format(fitting_utils.get_timestamp()))
	#plt.show()


def show_likelihood_trials_scatter(data_file_path):
	df = pd.read_csv(data_file_path)
	df = df[['subject', 'model', 'stage', 'day in stage', 'trial', 'likelihood']].copy()
	df.dropna(inplace=True)
	df = rename_models(df)

	# fig = plt.figure(figsize=(10, 5), dpi=100, facecolor='w')
	# for s, stage in enumerate(stages):
	# 	df_stage = df[df.stage==s+1]
	# 	axis = fig.add_subplot(1, 3, s+1)
	# 	sns.histplot(data=df_stage, x="likelihood", hue='model', stat='density', fill=True, binwidth=0.1,
	# 				  alpha=0.2, element="step", cumulative=True)
	# 	axis.legend([], [], frameon=False)
	#
	# 	handles, labels = axis.get_legend_handles_labels()
	# 	fig.legend(handles, labels, loc=(0.01, 0.90), prop={'size': 10},
	# 			   labelspacing=0.3)  # loc=(0.55,0.1), prop={'size': 7}


	sns.set_theme(style="white")
	fig = plt.figure(figsize=(11, 5), dpi=100, facecolor='w')

	for i, model in enumerate(stable_unique(df.model)):
		for s, stage in enumerate(stages):
			axis = fig.add_subplot(len(stable_unique(df.model)), 3, i * 3 + s + 1)
			model_df = df[(df.model == model) & (df.stage == s + 1)]
			sns.histplot(data=model_df, x="likelihood", kde=True, stat='percent', fill=True, element="step")
			axis.axvline(x=np.mean(model_df.likelihood), color='red', label='mean', alpha=0.7,)
			axis.axvline(x=np.median(model_df.likelihood), color='green', label='median', alpha=0.7)
			axis.axvline(x=0.5, color='gray', label='', alpha=0.7, linestyle='--')

			axis.set_xlabel(stages[s]) if i== len(stable_unique(df.model))-1 else axis.set_xlabel('')
			axis.set_xticklabels(['']) if i < len(stable_unique(df.model))-1 else 0
			axis.set_ylabel(model) if s == 0 else axis.set_ylabel('')

			axis.set_xlim([0,1])
			axis.set_ylim([0,15]) if s==1 else  axis.set_ylim([0,10])

			despine(axis)

	handles, labels = axis.get_legend_handles_labels()
	fig.legend(handles, labels, loc="upper right", prop={'size': 12}, labelspacing=0.3)  # loc=(0.55,0.1), prop={'size': 7}

	plt.subplots_adjust(left=0.07, bottom=0.12, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

	plt.savefig('fitting/Results/figures/trial_likelihood_dispersion_{}'.format(utils.get_timestamp()))
	#plt.show()


def plot_models_fitting_result_per_stage(data_file_path):
	df = pd.read_csv(data_file_path)
	df = rename_models(df)
	df['stage'] = df['stage'].astype('category')

	df = df[['subject', 'model', 'stage', 'day in stage', 'trial', 'likelihood']].copy()
	df['NLL'] = -np.log(df.likelihood)

	df_stage = df.groupby(['subject', 'model', 'stage'], sort=False).mean().reset_index()
	sns.set_theme(style="whitegrid")

	df['ML'] = np.exp(-df.NLL)
	df_stage['ML'] = np.exp(-df_stage.NLL)

	fig = plt.figure(figsize=(11, 3.5), layout="constrained")
	spec = fig.add_gridspec(1, 3)
	ax0 = fig.add_subplot(spec[0, 0:2])
	ax1 = fig.add_subplot(spec[0, 2])

	y = 'likelihood'
	g1 = sns.barplot(x='stage', y=y, hue='model', hue_order=models_order_df(df_stage),
					 data=df_stage, ax=ax0, errorbar='se' ,errwidth=1, capsize=.05)
	g1.set_ylim([0.5, 0.63])
	g1.set_xticklabels(stages)
	g1.set(xlabel='', ylabel='Average Likelihood')
	g1.legend([], [], frameon=False)

	args = dict(x="stage", y=y, hue="model", hue_order=models_order_df(df_stage))
	pairs = [((1, 'AARL'), (1, 'MFRL')), ((1, 'SARL'), (1, 'AARL')), ((1, 'FRL'), (1, 'MFRL')),
			 ((2, 'AARL'), (2, 'MFRL')), ((2, 'SARL'), (2, 'AARL')), ((2, 'FRL'), (2, 'MFRL')),
			 ((3, 'AARL'), (3, 'MFRL')), ((3, 'SARL'), (3, 'AARL')),((3, 'FRL'), (3, 'MFRL'))]

	annot = Annotator(g1, pairs, **args, data=df)
	annot.configure(test='t-test_paired', text_format='star', loc='inside', verbose=2)
	annot.apply_test().annotate()

	df['dummy'] = 1
	#g2 = sns.boxplot(x='dummy', y='ML', hue='model', data=df, ax=ax1)
	g2 = sns.barplot(x='dummy', y=y, hue='model', hue_order=models_order_df(df),
					 data=df, ax=ax1, errorbar='se', errwidth=1, capsize=.05)

	g2.set_xticklabels([''])
	g2.set(xlabel='All Stages', ylabel='')
	g2.legend([], [], frameon=False)
	plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.9, wspace=0.3, hspace=0.3)

	g2.set_ylim([0.5, 0.6])

	args = dict(x="dummy", y=y, hue="model", hue_order=models_order_df(df))
	pairs = [((1.0, 'AARL'), (1.0, 'MFRL')), ((1.0, 'SARL'), (1.0, 'AARL')), ((1.0, 'MFRL'), (1.0, 'FRL'))]

	annot = Annotator(g2, pairs, **args, data=df)
	annot.configure(test='t-test_paired', text_format='star', loc='inside', verbose=2)
	annot.apply_test().annotate()

	g2.set_ylim(g1.get_ylim())

	handles, labels = g2.get_legend_handles_labels()
	g2.legend(handles, labels, loc='lower left', prop={'size': 11}, labelspacing=0.2)

	plt.savefig('fitting/Results/figures/all_models_by_stage_{}'.format(utils.get_timestamp()))


def stage_transition_model_quality(data_file_path):
	df = pd.read_csv(data_file_path)
	df = df[['subject', 'model', 'stage', 'day in stage', 'likelihood']].copy()
	df = rename_models(df)
	df['NLL'] = -np.log(df.likelihood)

	model_df = df
	model_df['ML'] = np.exp(-model_df.NLL)

	transition1_before = model_df[(model_df['day in stage'] == 1) & (model_df.stage == 2)]
	transition1_end = model_df[(model_df.stage == 1)].groupby(['model', 'subject'], sort=False).max('day in stage').reset_index()
	transition1_df = pd.concat([transition1_before, transition1_end], ignore_index=True)

	transition2_before = model_df[(model_df['day in stage'] == 1) & (model_df.stage == 3)]
	transition2_end = model_df[(model_df.stage == 2)].groupby(['model', 'subject'], sort=False).max('day in stage').reset_index()
	transition2_df = pd.concat([transition2_before, transition2_end], ignore_index=True)

	fig = plt.figure(figsize=(10, 5), layout="constrained")

	axis1 = fig.add_subplot(121)
	axis2 = fig.add_subplot(122)
	g1 = sns.pointplot(x='stage', y='likelihood', hue='model', errorbar='se', data=transition1_df, ax=axis1)
	plt.setp(g1.collections, alpha=.5)  # for the markers
	plt.setp(g1.lines, alpha=.5)  # for the lines

	g2= sns.pointplot(x='stage', y='likelihood', hue='model', errorbar='se', data=transition2_df, ax=axis2)
	axis1.legend([], [], frameon=False), axis2.legend([], [], frameon=False)
	plt.setp(g2.collections, alpha=.5)  # for the markers
	plt.setp(g2.lines, alpha=.5)  # for the lines

	axis1.set_ylim(0.45, 0.95)
	axis2.set_ylim(0.45, 0.95)
	handles, labels = axis2.get_legend_handles_labels()
	fig.legend(handles, labels, loc='upper right', prop={'size': 11})

	axis1.axvline(x=0.5, ymin=0.05, ymax=0.95, alpha=0.5, dashes=(5, 2, 1, 2), lw=2, zorder=0, clip_on=False, color='black')
	axis2.axvline(x=0.5, ymin=0.05, ymax=0.95, alpha=0.5, dashes=(5, 2, 1, 2), lw=2, zorder=0, clip_on=False, color='black')

	axis1.axhline(y=0.5, alpha=0.7, lw=1, color='grey', linestyle='--')
	axis2.axhline(y=0.5, alpha=0.7, lw=1, color='grey', linestyle='--')

	axis1.set_xlabel('')
	axis2.set_xlabel('')
	axis1.set_ylabel('Average Likelihood')
	axis2.set_ylabel('')

	axis1.set_xticklabels(stages[:2])
	axis2.set_xticklabels(stages[1:])

	despine(axis1)
	despine(axis2)

	plt.savefig('fitting/Results/figures/stage_transition_{}'.format(fitting_utils.get_timestamp()))

# # plots.plot_histogram(result=brain_results, dimension_identifier='lr', bins=20)
# # plots.plot_objective_2D(brain_results['results'], 'lr', 'batch_size')
# # plots.plot_objective(brain_results['results'], plot_dims=['nmr', 'lr'])


def compare_fitting_criteria(data_file_path):
	df = pd.read_csv(data_file_path)
	df = df[['subject', 'model', 'likelihood', 'parameters', 'day in stage', 'stage', 'reward']].copy()
	df = rename_models(df)
	df['LL'] = np.log(df.likelihood)

	# # optimization average over trials
	likelihood_trial = df.groupby(['model']).agg({'reward': 'count', 'LL': 'sum', 'likelihood': 'mean'}).reset_index()
	data = likelihood_trial.rename(columns={'reward': 'n'})

	#data['k'] = data.apply(lambda row: len(fitting_utils.string2list(row['parameters'])), axis=1)
	data['k'] = data.apply(lambda row: 3 if row.model=='AARL' else 2, axis=1)

	data['AIC'] = - 2 * data.LL/data.n + 2 * data.k/data.n
	data['BIC'] = - 2 * data.LL + np.log(data.n) * data.k
	data['LPT'] = data.likelihood

	data.LL = -data.LL
	for criterion in ['AIC', 'BIC', 'LPT']:
		# fig = plt.figure(figsize=(35, 7), dpi=120, facecolor='w')
		# for subject in stable_unique(data.subject):
		# 	axis = fig.add_subplot(3, 3, subject + 1)
		# 	subject_model_df = data[(data.subject == subject)]
		# 	sns.barplot(x=criterion, y='model', data=subject_model_df, ax=axis, orient='h', order=models_order)
		# 	axis.set_title('Subject:{}'.format(subject+1))
		# 	minn = np.min(subject_model_df[criterion])
		# 	maxx=np.max(subject_model_df[criterion])
		# 	delta = 0.1*(maxx-minn)
		# 	axis.set_xlim([minn-delta,maxx+delta])
		# 	labels = axis.get_xticklabels()
		# 	axis.set_ylabel("")
		# 	axis.set_yticklabels("") if subject % 3 > 0 else 0
		# 	axis.set_xlabel("") if subject < 6 else 0
		#
		# plt.subplots_adjust(left=0.15, bottom=0.1, right=0.97, top=0.9, wspace=0.2, hspace=0.4)

		#plot the average fitting quality for the entire population.
		#sum_df = likelihood_trial.groupby(['model']).mean().reset_index()

		plt.figure(figsize=(4.5, 4), dpi=120, facecolor='w')
		axis = sns.barplot(x='model', y=criterion, data=data, order=models_order_df(data)) #orient='v'
		minn = np.min(data[criterion])
		maxx = np.max(data[criterion])
		delta = 0.1 * (maxx - minn)
		plt.ylim([minn - delta, maxx + delta])

		plt.subplots_adjust(left=0.21, bottom=0.1, right=0.97, top=0.95, wspace=0.2, hspace=0.4)
		axis.spines['top'].set_visible(False)
		axis.spines['right'].set_visible(False)

		axis.set_xlabel('')

		plt.savefig('fitting/Results/figures/{}_{}'.format(criterion, utils.get_timestamp()))


def show_fitting_parameters(data_file_path):
	df_all = pd.read_csv(data_file_path)
	df = df_all[['subject', 'model', 'parameters', 'likelihood']].copy()
	df = rename_models(df)
	df = df.groupby(['subject', 'model', 'parameters'], sort=False).mean().reset_index()
	k = df.parameters.apply(lambda row: fitting_utils.string2list(row))
	parameters = k.apply(pd.Series)
	df = df.join(parameters)
	df = df.rename(columns={0: "beta", 1: "alpha", 2: "alpha_phi"})
	df['subject'] = df['subject'].astype('category')

	param_mean = df.groupby(['model']).mean().reset_index()
	param_std = df.groupby(['model']).sem().reset_index()

	params_info = param_mean.merge(param_std, on=['model'])
	params_info = params_info.sort_values(['model'], ascending=False)

	params_info['alpha'] = params_info.apply(lambda row: "${:.2} \\pm {:.2}$".format(row.alpha_x, row.alpha_y),
													 axis=1)
	params_info['beta'] = params_info.apply(lambda row: "${:.2} \\pm {:.2}$".format(row.beta_x, row.beta_y),
													 axis=1)
	params_info['alpha_phi'] = params_info.apply(lambda row: "${:.2} \\pm {:.2}$".format(row.alpha_phi_x, row.alpha_phi_y),
													 axis=1)
	params_info = params_info[['model', 'alpha', 'beta', 'alpha_phi' ]]
	print(params_info)

	# ax = sns.scatterplot(data=df, x='alpha', y='alpha_phi', hue='model')
	# ax.set_xlim([-0.01, 0.1])
	# ax.set_ylim([-0.01, 0.1])
	# ax = sns.pairplot(hue='model', data=df, diag_kind="hist")
	# ax.set(xscale="log", yscale="log")


def model_parameters_development(data_file_path):

	reported_days_in_stage3 = 9
	plt.rcParams.update({'font.size': 14})
	df_all = pd.read_csv(data_file_path)
	df = df_all[['subject', 'model', 'parameters', 'stage', 'day in stage', 'trial', 'model_variables', 'likelihood']].copy()

	df = rename_models(df)
	#format the model_variables entry
	df = df[df['model_variables'].notna()].reset_index()

	#remove irrelevant trials
	df = df[~((df.stage == 1) & (df['day in stage'] > num_days_reported[0]))]
	df = df[~((df.stage == 2) & (df['day in stage'] > num_days_reported[1]))]
	df = df[~((df.stage == 3) & (df['day in stage'] > reported_days_in_stage3))]

	for model in ['AARL']:
		df_model = df[df.model == model]
		df_model['model_variables'] = df_model['model_variables'].apply(lambda s: s.replace("\'", "\""))
		df_model['model_variables'] = df_model['model_variables'].apply(json.loads)

		variables_names = df_model['model_variables'].tolist()[0].keys()
		df_variables = pd.DataFrame(df_model['model_variables'].tolist())
		df_no = df_model.drop('model_variables', axis=1).reset_index()
		df_model = pd.concat([df_no, df_variables], axis=1)

		df_model = df_model.groupby(['subject', 'model', 'parameters', 'stage', 'day in stage'],
									sort=False).mean().reset_index()

		df_model['ind'] = df_model.stage + 0.1 * df_model['day in stage']
		df_model['ind'] = df_model['ind'].astype(str)

		fig = plt.figure(figsize=(7.5, 4), dpi=120, facecolor='w')

		sns.set_palette("Set2", n_colors=3)
		axis = fig.add_subplot(111)
		for variable_name in variables_names:
			sns.lineplot(x="ind", y=variable_name, data=df_model, errorbar="se", err_style='band', ax=axis, label=variable_name.split('_')[0], marker='o')
		axis.legend(loc='upper left')

		for stage_day in [3, 5]:
			plt.axvline(x=stage_day + 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=1, color='gray')

		plt.xlabel('Stage.Day')
		plt.ylabel('Attention')

		handles, labels = axis.get_legend_handles_labels()
		plt.legend(handles, labels, loc="upper left", prop={'size': 16}, labelspacing=0.2)

		plt.subplots_adjust(left=0.12, bottom=0.15, right=0.97, top=0.9, wspace=0.2, hspace=0.4)

		axis.spines['top'].set_visible(False)
		axis.spines['right'].set_visible(False)

		plt.savefig('fitting/Results/paper_figures/attention_{}'.format(fitting_utils.get_timestamp()))

		plt.rcParams.update({'font.size': 10})
		fig = plt.figure(figsize=(9, 7), dpi=120, facecolor='w')
		for i, subject in enumerate(stable_unique(df_model["subject"])):

			df_sub = df_model[df_model["subject"] == subject]
			axis = fig.add_subplot(4, 2, i+1)

			for variable_name in variables_names:
				axis = sns.lineplot(x="ind", y=variable_name, data=df_sub, errorbar="se", err_style='band', ax=axis, label=variable_name.split('_')[0])
			for stage_day in [3, 5]:
				axis.axvline(x=stage_day + 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2, color='gray')
			axis.legend([], [], frameon=False)
			axis.spines['top'].set_visible(False)
			axis.spines['right'].set_visible(False)

			axis.set_xlabel('Stage.Day') if i > 5 else axis.set_xlabel('')
			axis.set_ylabel('') #axis.set_ylabel("Attention") if i % 2 == 0 else axis.set_ylabel('')
			plt.tick_params(axis='both', which='major', labelsize=9)

	plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.95, wspace=0.1, hspace=0.4)

	handles, labels = axis.get_legend_handles_labels()
	fig.legend(handles, labels, loc=(0.1,0.9), prop={'size': 11}, labelspacing=0.2)

	x = 1


def average_likelihood(data_file_path):
	df = pd.read_csv(data_file_path)
	data = df[['subject', 'model', 'likelihood', 'day in stage', 'trial', 'stage', 'reward']].copy()
	data = data.groupby(['subject', 'model'], sort=['likelihood']).mean().reset_index()
	data = rename_models(data)

	# renumber the subject to reflect the best likelihood
	order = data.groupby('subject').max('likelihood').sort_values('likelihood', ascending=True).reset_index()
	subject_map = order.subject.to_dict()
	# invert the dictionary and add numbering from 1
	subject_map = {v: k+1 for k, v in subject_map.items()}
	data.subject = data.subject.map(lambda x: subject_map[x])

	criterion = 'likelihood'
	plt.figure(figsize=(8, 4), dpi=120, facecolor='w')

	axis = sns.scatterplot(x='subject', y=criterion, hue='model', alpha=0.7, data=data,
						   hue_order=models_order_df(data))
	plt.ylim([0.55, 0.68])

	plt.subplots_adjust(left=0.15, bottom=0.15, right=0.97, top=0.95, wspace=0.2, hspace=0.4)
	despine(axis)

	axis.set_xlabel('Animal')
	axis.set_ylabel('Average Likelihood')

	handles, labels = axis.get_legend_handles_labels()
	axis.legend(handles, labels, loc="upper left", prop={'size': 11},
			   labelspacing=0.3)

	colors = sns.color_palette()
	averages = data.groupby(['model'], sort=False).mean().reset_index()
	averages = averages[['model', 'likelihood']]
	#for ind, (model, likelihood) in enumerate(list(averages.itertuples(index=False, name=None))):
	for ind, model in enumerate(models_order_df(data)):
		likelihood = averages[averages.model==model].likelihood.values[0]
		axis.axhline(y=likelihood, alpha=1, lw=2, color=colors[ind])

if __name__ == '__main__':
	file_path = 'fitting/Results/Rats-Results/reported_results_tabular_modelling/main_results_reported_10_1.csv' #reported
	#learning_curve_behavioral_boxplot('fitting/Results/Rats-Results/reported_results_tabular_modelling/all_data.csv')
	show_days_to_criterion('fitting/Results/Rats-Results/reported_results_dimensional_shifting/all_data.csv')
	# models_fitting_quality_over_times_average(file_path)
	#models_fitting_quality_over_times(file_path)
	# compare_model_subject_learning_curve_average(file_path)
	##compare_model_subject_learning_curve(file_path)
	# plot_models_fitting_result_per_stage(file_path)
	# show_likelihood_trials_scatter(file_path)
	#stage_transition_model_quality(file_path)
	# show_fitting_parameters(file_path)
	#compare_fitting_criteria(file_path)
	# average_likelihood(file_path)
	#compare_neural_tabular_models(file_path)
	# model_parameters_development(file_path)
	x = 2
