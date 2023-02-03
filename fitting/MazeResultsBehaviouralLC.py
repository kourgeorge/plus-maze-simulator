import functools
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin
import scipy
import seaborn as sns
import statsmodels.api as sm
from statannotations.Annotator import Annotator
from statsmodels.formula.api import ols

import utils
from fitting import fitting_utils
from fitting.fitting_utils import stable_unique, rename_models, models_order_df

plt.rcParams.update({'font.size': 12})

#stages = ['ODOR1', 'ODOR2', 'LED']
stages = ['Initial', 'IDS', 'MS', 'MS+IDS', 'EDS (Spatial)']
num_days_reported = [8, 2, 4, 2, 2]


def triplet_colors(n):
	pastel = sns.color_palette("pastel", n)
	deep = sns.color_palette("deep", n)
	dark = sns.color_palette("dark", n)

	colors = utils.flatten_list(list(zip(pastel, deep, dark)))
	return colors

def one_way_anova(df, target, c):
	model = ols('{} ~ C({})'.format(target,c), data=df).fit()
	result = sm.stats.anova_lm(model, type=2)
	print(result)


def RM_anova(df, depvar, subjects, within):
	#results = AnovaRM(data=df, depvar=depvar, subject=subjects, within=[within]).fit() #, aggregate_func='mean'
	aov = pingouin.rm_anova(data=df, dv=depvar, subject=subjects, within=within,  detailed=True)
	aov.round(3)

	print(aov)


def two_way_anova(df, target, c1, c2):
	model = ols('{} ~ C({}) + C({}) +\
	C({}):C({})'.format(target,c1,c2,c1,c2),
				data=df).fit()
	result = sm.stats.anova_lm(model, type=2)
	print(result)

def despine(axis):
	axis.spines['top'].set_visible(False)
	axis.spines['right'].set_visible(False)


def dilute_xticks(axis, k=2):
	ticks = ["{}".format(int(x._text[2:])) if (int(x._text[2:]) - 1) % k == 0 else "" for ind, x in enumerate(axis.get_xticklabels())]
	axis.set_xticklabels(ticks)

def filter_days(df):
	for ind, stage in enumerate(stages):
		df = df[~((df.stage == ind+1) & (df['day in stage'] > num_days_reported[ind]))]
	return df

def extract_model_name_type(df):
	df['model_type'] = df.model.apply(lambda x: x.split('-')[-1])
	df['model_struct'] = df.model.apply(lambda x: 'm' if len(x.split('-')) == 1 else '-'.join(x.split('-')[:-1])+'-m')
	return df


def index_days(df):
	df['ind'] = df.apply(lambda x:str(x.stage)+'.'+str(x['day in stage']), axis='columns')

	def compare(x, y):
		if int(x[0]) < int(y[0]):
			return -1
		elif int(x[0]) > int(y[0]):
			return 1
		elif int(x[2:]) < int(y[2:]):
			return -1
		else:
			return 1

	order = sorted(np.unique(df['ind']), key=functools.cmp_to_key(compare))
	transition = [i for i in range(1, len(order)) if order[i][0] != order[i - 1][0]]

	return df, order, transition


def calculate_goal_choice(df):
	df = filter_days(df)

	df = df[['subject', 'model', 'stage', 'day in stage', 'trial', 'action', 'reward', 'initial_motivation']].copy()
	df['reward_type'] = df['action'].map(lambda x: 'Food' if x in [1, 2] else 'Water')
	data = df.groupby(['model', 'subject', 'stage', 'day in stage', 'reward_type','initial_motivation'], sort=False). \
		agg({'trial': 'count'}).reset_index()
	

	dd = data.pivot(index=['model', 'subject', 'stage', 'day in stage', 'initial_motivation'], columns='reward_type').reset_index()
	dd.columns = [' '.join(col).strip() for col in dd.columns.values]
	dd['gc'] = dd.apply(lambda row: (row['trial Water'] - row['trial Food']) / (row['trial Water'] + row['trial Food']) if row.initial_motivation == 'water' else
								 (row['trial Food'] - row['trial Water']) / (row['trial Food'] + row['trial Water']), axis=1)

	return dd

def sort_subject_by_criterion(data, criterion):
	data = data.groupby(['subject', 'model'], sort=[criterion]).mean().reset_index()
	# renumber the subject to reflect the best likelihood
	order = data.groupby('subject').max(criterion).sort_values(criterion, ascending=True).reset_index()
	subject_map = order.subject.to_dict()
	# invert the dictionary and add numbering from 1
	subject_map = {v: k+1 for k, v in subject_map.items()}
	data.subject = data.subject.map(lambda x: subject_map[x])
	return data


def unbox_parameters(df, params):

	df = df.groupby(['subject', 'model', 'parameters'], sort=False).mean().reset_index()
	k = df.parameters.apply(lambda row: fitting_utils.string2list(row))
	parameters = k.apply(pd.Series)
	df = df.join(parameters)
	df = df.rename(columns={k: v for k, v in enumerate(params)})
	df['subject'] = df['subject'].astype('category')

	return df


def unbox_model_variables(df_model):
	# format the model_variables entry
	df_model['model_variables'] = df_model['model_variables'].apply(lambda s: s.replace("\'", "\""))
	df_model['model_variables'] = df_model['model_variables'].apply(json.loads)

	variables_names = df_model['model_variables'].tolist()[0].keys()
	df_variables = pd.DataFrame(df_model['model_variables'].tolist())

	df_no = df_model.drop('model_variables', axis=1).reset_index()
	df_model = pd.concat([df_no, df_variables], axis=1)

	# df_model = df_model.groupby(['subject', 'model', 'parameters', 'stage', 'day in stage', 'ind'],
	# 							sort=False).mean().reset_index()

	return df_model, variables_names

####################################################################
###################### Behavioural Statistics ######################


def goal_choice_index(data_file_path, simulations=False):
	plt.rcParams.update({'font.size': 16})
	df = pd.read_csv(data_file_path)

	if not simulations:
		df = df[df.model==df.model[0]]

	dd = calculate_goal_choice(df)
	dd, order, tr = index_days(dd)

	fig = plt.figure(figsize=(10, 5))
	axis = sns.pointplot(x="ind", y="gc", data=dd, errorbar="se", join=False, order=order,
						 capsize=.3, scale=1.5, dodge=0.2, color='black', linestyles='--')

	for stage_day in tr:
		axis.axvline(x=stage_day - 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2, color='gray')

	axis.axhline(y=0, alpha=0.5, lw=2, color='gray', linestyle='--')

	axis.set_ylim([-1, 1])
	despine(axis)
	dilute_xticks(axis,2)
	axis.set_ylabel('Goal Choice Index')
	axis.set_xlabel('Training Day in Stage')

	plt.subplots_adjust(left=0.12, bottom=0.15, right=0.99, top=0.95, wspace=0.1, hspace=0.4)


def show_days_to_criterion(data_file_path):
	df = pd.read_csv(data_file_path)

	df = df[df.model == df.model[0]]
	df = df[['subject', 'stage', 'day in stage', 'model', 'trial', 'reward']].copy()

	df = rename_models(df)
	df = fitting_utils.cut_off_data_when_reaching_criterion(df, num_stages=5)
	df = df.groupby(['subject', 'model', 'stage'], sort=False).agg({'day in stage': 'max'}).reset_index()

	RM_anova(df, 'day in stage', 'subject', within='stage')

	fig = plt.figure(figsize=(10, 5))
	g1 = sns.barplot(x='stage', y='day in stage', order=list(range(1, len(stages) + 1)),
					 fill=False, data=df, errorbar='se', errwidth=1, capsize=.05)

	g1.set(xlabel='', ylabel='Days Until Criterion')

	pairs = [((2), (3)), ((3), (4)),
			 ((3), (5)), ((1), (2)), ((1), (3)), ((1), (4)), ((1), (5))]
	annot = Annotator(g1, pairs, x='stage', y='day in stage', data=df)
	annot.configure(test='t-test_paired', text_format='star', loc='inside', verbose=2, line_height=0.05,
					comparisons_correction="Bonferroni")
	annot.apply_test().annotate()
	g1.set_xticklabels(stages)
	despine(g1)
	plt.subplots_adjust(left=0.08, bottom=0.1, right=0.99, top=0.95, wspace=0.1, hspace=0.4)

	RM_anova(df, 'day in stage', 'subject', within='stage')

	x = 1
	

def water_food_correct(data_file_path, initial_motivation='water'):
	plt.rcParams.update({'font.size': 16})
	df = pd.read_csv(data_file_path)
	df = df[['subject', 'model', 'stage', 'day in stage', 'trial', 'reward', 'action', 'initial_motivation']].copy()

	model = np.unique(df.model)[2]
	df = df[df.model == model]

	df = df[df.initial_motivation == initial_motivation]

	df['reward_type'] = df['action'].map(lambda x: 'Food' if x in [1,2] else 'Water')

	days_info_df = df.groupby(['subject', 'model', 'stage', 'day in stage', 'reward_type'], sort=False). \
		agg({'reward': 'mean', 'trial': 'count'}).reset_index()

	days_info_df = filter_days(days_info_df)
	days_info_df, order, tr = index_days(days_info_df)
	days_info_df.sort_values('ind', axis=0, ascending=True, inplace=True)

	colors = triplet_colors(5)
	colors =(colors[9], colors[10])
	colors = sns.color_palette("Greys", n_colors=2)
	colors = (colors[0], colors[1])

	fig = plt.figure(figsize=(10, 5))
	axis = sns.pointplot(x="ind", y="reward", hue="reward_type", data=days_info_df, errorbar="se", join=False, capsize=.2,
						 palette=sns.color_palette(colors), scale=0.7, dodge=0.2, linestyles='--')

	for stage_day in tr:
		axis.axvline(x=stage_day - 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2, color='gray')

	despine(axis)
	axis.legend().set_title('')

	axis.set_xlabel('Training Day in Stage')
	axis.set_ylabel('Correct Choices')

	df_temp = df[['subject', 'model', 'stage', 'day in stage', 'trial']].copy()
	trials_in_day = df_temp.groupby(['subject', 'model', 'stage', 'day in stage'], sort=False).count().reset_index()
	trials_in_day['total_trials'] = trials_in_day['trial']
	trials_in_day.drop(['trial'], axis=1, inplace=True)

	dilute_xticks(axis,2)

	for stage_day in tr:
		axis.axvline(x=stage_day - 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2, color='gray')

	axis.axhline(y=0.5, alpha=0.7, lw=1, color='grey', linestyle='--')
	axis.axhline(y=0.75, alpha=0.7, lw=1, color='grey', linestyle='--')

	plt.subplots_adjust(left=0.08, bottom=0.15, right=0.99, top=0.99, wspace=0.1, hspace=0.4)


def learning_curve_behavioral_boxplot(data_file_path):
	plt.rcParams.update({'font.size': 16})
	df = pd.read_csv(data_file_path)
	df = df[['subject', 'model', 'stage', 'day in stage', 'trial', 'reward', 'model_reward']].copy()
	df = df[df.model == df.model[0]]
	days_info_df = df.groupby(['subject', 'model', 'stage', 'day in stage'], sort=False).mean().reset_index()

	days_info_df, order, st = index_days(days_info_df)

	fig = plt.figure(figsize=(10, 5))
	axis = sns.boxplot(data=days_info_df, x='ind', y='reward',  order=order, color='grey', width=0.65, fliersize=4)#, palette="crest")

	for stage_day in st:
		axis.axvline(x=stage_day - 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2, color='grey')

	animals_in_day = [len(np.unique(days_info_df[days_info_df.ind==day_stage].subject)) for day_stage in order]
	axis.axhline(y=0.5, alpha=0.7, lw=1, color='grey', linestyle='--')
	axis.axhline(y=0.75, alpha=0.7, lw=1, color='grey', linestyle='--')

	#add count of animals in each day.
	axis.set_ylabel([0.3, 1])
	for xtick in axis.get_xticks():
		axis.text(xtick, 0.26, animals_in_day[xtick],
				horizontalalignment='center',size='small',color='black',weight='semibold')

	ticks = ["{}".format(int(x[2:])) if (int(x[2:])-1) % 2 == 0 else "" for ind, x in enumerate(order) ]
	axis.set_xticklabels(ticks)

	axis.set_xlabel('Training Day in Stage')
	axis.set_ylabel('Correct Choice Rate')

	axis.spines['top'].set_visible(False)
	axis.spines['right'].set_visible(False)

	plt.subplots_adjust(left=0.08, bottom=0.15, right=0.99, top=0.99, wspace=0.1, hspace=0.4)

#################### Daily Figures #####################

def models_fitting_quality_over_times_average(data_file_path, models=None, palette=None):
	plt.rcParams.update({'font.size': 16})
	df = pd.read_csv(data_file_path)
	df = df[['subject', 'model', 'stage', 'day in stage', 'trial', 'likelihood', 'reward', 'model_reward']].copy()
	df['NLL'] = -np.log(df.likelihood)

	model_df = df.groupby(['subject', 'model', 'stage', 'day in stage'], sort=False).mean().reset_index()
	model_df['ML'] = np.exp(-model_df.NLL)

	model_df = rename_models(model_df)
	if models:
		model_df = model_df[model_df.model.isin(models)]

	model_df = filter_days(model_df)
	model_df, order, tr = index_days(model_df)
	# this is needed because there is no good way to order the x-axis in lineplot.
	model_df.sort_values('ind', axis=0, ascending=True, inplace=True)

	if palette:
		sns.set_palette(palette)
	else:
		sns.set_palette("colorblind", len(models_order_df(model_df)))


	fig = plt.figure(figsize=(7.5, 4), dpi=100, facecolor='w')
	# colors = triplet_colors(3)
	# sns.set_palette([colors[6],colors[8]])

	axis = sns.lineplot(x="ind", y="likelihood", hue="model", size_order=order.reverse(), sort=False,
						hue_order=models_order_df(model_df),
						data=model_df, errorbar="se", err_style='bars')

	for stage_day in tr:
		axis.axvline(x=stage_day - 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2, color='gray')

	axis.axhline(y=0.25, alpha=0.5, linestyle='--' , lw=1, color='gray')

	axis.set_xlabel('Training Day in stage')
	axis.set_ylabel('Average Likelihood')

	handles, labels = axis.get_legend_handles_labels()
	plt.legend(handles, labels, loc="upper left", prop={'size': 14}, labelspacing=0.2)
	despine(axis)
	dilute_xticks(axis, k=2)
	plt.subplots_adjust(left=0.12, bottom=0.15, right=0.98, top=0.98, wspace=0.2, hspace=0.1)

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


def compare_model_subject_learning_curve_average(data_file_path, models=None):
	plt.rcParams.update({'font.size': 18})
	df = pd.read_csv(data_file_path)

	df = rename_models(df)

	df = df[['subject', 'model', 'stage', 'day in stage', 'trial', 'reward', 'model_reward']].copy()
	model_df = df.groupby(['subject', 'model', 'stage', 'day in stage'], sort=False).mean().reset_index()

	model_df = filter_days(model_df)

	if models:
		model_df = model_df[model_df.model.isin(models)]

	model_df, order, tr = index_days(model_df)
	# this is needed because there is no good way to order the x-axis in lineplot.
	model_df.sort_values('ind', axis=0, ascending=True, inplace=True)


	fig = plt.figure(figsize=(7.5, 4), dpi=100, facecolor='w')
	sns.set_palette(sns.color_palette('colorblind'))
	axis = sns.lineplot(x="ind", y="model_reward", hue="model", hue_order=models_order_df(model_df),
						data=model_df, errorbar="se", err_style='band')

	subject_reward_df = model_df[['ind','subject', 'stage', 'day in stage', 'trial', 'reward']].copy().drop_duplicates()
	axis = sns.lineplot(x="ind", y="reward", data=subject_reward_df, errorbar="se", err_style='band', linestyle='--',
						ax=axis, color='black')

	for stage_day in tr:
		axis.axvline(x=stage_day - 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2, color='gray')

	axis.set_xlabel('Training day in stage')
	axis.set_ylabel('Success Rate')

	handles, labels = axis.get_legend_handles_labels()
	plt.legend(handles, labels, loc="lower right", prop={'size': 18}, labelspacing=0)
	despine(axis)
	dilute_xticks(axis,2)
	plt.subplots_adjust(left=0.12, bottom=0.17, right=0.98, top=0.98, wspace=0.2, hspace=0.1)

	plt.savefig('fitting/Results/paper_figures/learning_curve_{}'.format(utils.get_timestamp()))


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
			axis = fig.add_subplot(len(stable_unique(df.model)), len(stages), i * len(stages) + s + 1)
			model_df = df[(df.model == model) & (df.stage == s + 1)]
			sns.histplot(data=model_df, x="likelihood", kde=True, stat='percent', fill=True, element="step")
			axis.axvline(x=np.mean(model_df.likelihood), color='red', label='mean', alpha=0.7,)
			axis.axvline(x=np.median(model_df.likelihood), color='green', label='median', alpha=0.7)
			axis.axvline(x=0.25, color='gray', label='', alpha=0.7, linestyle='--')

			axis.set_xlabel(stages[s]) if i== len(stable_unique(df.model))-1 else axis.set_xlabel('')
			axis.set_xticklabels(['']) if i < len(stable_unique(df.model))-1 else 0
			axis.set_ylabel(model) if s == 0 else axis.set_ylabel('')

			#axis.set_xlim([0,1])
			#axis.set_ylim([0,15]) if s==1 else  axis.set_ylim([0,10])

			despine(axis)

	handles, labels = axis.get_legend_handles_labels()
	fig.legend(handles, labels, loc="upper right", prop={'size': 12}, labelspacing=0.3)  # loc=(0.55,0.1), prop={'size': 7}

	plt.subplots_adjust(left=0.07, bottom=0.12, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

	plt.savefig('fitting/Results/figures/trial_likelihood_dispersion_{}'.format(utils.get_timestamp()))
	#plt.show()

####################Fitting Criteria ############################

def plot_models_fitting_result_per_stage_action_bias(data_file_path):
	plt.rcParams.update({'font.size': 16})
	df = pd.read_csv(data_file_path)
	df = rename_models(df)
	relevant_models= utils.flatten_list([(m,'B-'+m,'M(B)-'+m) for m in ['SARL','ORL','FRL']])
	df=df[df.model.isin(relevant_models)]


	df = df[['subject', 'model', 'stage', 'day in stage', 'trial', 'likelihood']].copy()
	df = df.groupby(['subject', 'model', 'stage', 'day in stage']).mean().reset_index()

	df['stage'] = df['stage'].astype('category')

	sns.set_theme(style="white")

	models = ['SARL','ORL','FRL']
	n=len(models)

	fig = plt.figure(figsize=(12, 5))
	spec = fig.add_gridspec(n, 4)
	ax0 = fig.add_subplot(spec[0:n, 3])
	ax = [None, None, None]

	colors = triplet_colors(3)
	ylim = [0.28,0.5]

	y = 'likelihood'
	for ind, model in enumerate(models):
		ax[ind] = fig.add_subplot(spec[ind, 0:n])
		sns.set_palette(colors[n*ind:n*ind+n])
		df_model = df[df.model.str.contains(model)]
		hue_order = [model_name for model_name in relevant_models if model in model_name]
		args = dict(x='stage', y=y, hue='model', data=df_model, hue_order=hue_order)
		ax[ind] = sns.barplot(**args, ax=ax[ind], errorbar='se' ,errwidth=1, capsize=.07, errcolor='gray', dodge=1)

		ax[ind].set_ylim(0.27, 0.38)

		pairs = [[((loc, model), (loc, 'B-'+model)), ((loc, model), (loc, 'M(B)-'+model))] for loc in range(1,6)]
		pairs = utils.flatten_list(pairs)
		annot = Annotator(ax[ind], pairs, **args)

		annot.configure(test='t-test_paired', text_format='star', loc='inside', verbose=2, line_height=0.05,
						comparisons_correction="bonferroni", line_offset=0)

		annot.apply_test().annotate()


		ax[ind].set(xlabel='', ylabel=model)
		ax[ind].legend([], [], frameon=False)
		ax[ind].set(xticklabels=[])
		ax[ind].tick_params(bottom=False)

		despine(ax[ind])

	ax[n-1].set_xticklabels(stages)
	fig.text(0.01, 0.5, 'Average Likelihood', va='center', rotation='vertical')

	sns.set_palette(sns.color_palette('Greys', n_colors=n))

	df = extract_model_name_type(df)
	ax0 = sns.barplot(x='model_type', y=y, hue='model_struct', hue_order=['m', 'B-m', 'M(B)-m'], #models_order_df(df),
					 data=df, ax=ax0, errorbar='se', errwidth=1, capsize=.1)

	ax0.set_xlabel('')
	ax0.legend([], [], frameon=False)

	ax0.set_ylim(ax[0].get_ylim())

	args = dict(x="model_type", y=y, hue="model_struct", hue_order=['m', 'B-m', 'M(B)-m'])
	pairs = utils.flatten_list([[((model, 'm'), (model, 'M(B)-m')),((model, 'B-m'), (model, 'm')),((model, 'B-m'), (model, 'M(B)-m'))] for model in models])
	annot = Annotator(ax0, pairs, **args, data=df)
	annot.configure(test='t-test_paired', text_format='star', loc='inside', line_height=0.01, verbose=1,
					comparisons_correction="bonferroni")
	annot.apply_test().annotate()

	handles, labels = ax0.get_legend_handles_labels()
	ax0.legend(handles, labels, loc='upper left', prop={'size': 12}, labelspacing=0)
	despine(ax0)
	ax0.set_ylabel('')

	fig.subplots_adjust(left=0.08, bottom=0.07, right=0.99, top=0.97, wspace=0.3, hspace=0.3)

	plt.savefig('fitting/Results/figures/all_models_by_stage_{}'.format(utils.get_timestamp()))


def compare_fitting_criteria(data_file_path, models=None, pallete=None):
	plt.rcParams.update({'font.size': 14})

	df = pd.read_csv(data_file_path)
	df = df[['subject', 'model', 'likelihood', 'parameters', 'day in stage', 'stage', 'reward']].copy()
	df = rename_models(df)

	if models:
		df = df[df.model.isin(models)]
	else:
		models = np.unique(df.model)

	df = extract_model_name_type(df)

	df['LL'] = np.log(df.likelihood)
	# # optimization average over trials
	likelihood_trial = df.groupby(['subject', 'model', 'model_type', 'model_struct', 'parameters'])\
		.agg({'reward': 'count', 'LL': 'sum', 'likelihood': 'mean'}, sort=False).reset_index()
	data = likelihood_trial.rename(columns={'reward': 'n'})

	#data['k'] = data.apply(lambda row: len(fitting_utils.string2list(row['parameters'])), axis=1)
	data['k'] = data.apply(lambda row: 3 if row.model in ['SARL','ORL','FRL', 'M(V)-SARL','M(V)-ORL','M(V)-FRL'] else 4, axis=1)

	data['ALPT'] = data.likelihood
	data['AvgNLL'] = -data.LL/data.n

	sum_df = data.groupby(['model']).agg({'LL':'sum', 'k':'mean','n':'sum'}).reset_index()
	
	sum_df['AIC']= -2 * sum_df.LL + 2 * sum_df.k.astype(int)
	sum_df['BIC'] = -2 * sum_df.LL + np.log(sum_df.n) * sum_df.k.astype(int)
	#sum_df = data
	if pallete:
		sns.set_palette(pallete)
	elif len(models) %3 ==0:
		sns.set_palette(triplet_colors(3))
	else:
		sns.set_palette("colorblind")

	for criterion in ['AIC', 'BIC']:

		plt.figure(figsize=(4.5, 4), dpi=120, facecolor='w')
		#axis = sns.barplot(y='model_type', x=criterion, hue='model_struct', data=sum_df)#, order=models_order_df(sum_df),)
		axis = sns.barplot(y='model', x=criterion, data=sum_df, order=models_order_df(sum_df), )
		minn = np.min(sum_df[criterion])
		maxx = np.max(sum_df[criterion])
		delta = 0.1 * (maxx - minn)
		plt.xlim([minn - delta, maxx + delta])

		despine(axis)

		axis.set_ylabel('')
		#axis.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

		if len(models)==9:
			axis.axhline(y=2 + 0.5, alpha=0.7, lw=1, color='grey', linestyle='-')
			axis.axhline(y=5 + 0.5, alpha=0.7, lw=1, color='grey', linestyle='-')
		elif len(models)==6:
			axis.axhline(y=1 + 0.5, alpha=0.7, lw=1, color='grey', linestyle='-')
			axis.axhline(y=3 + 0.5, alpha=0.7, lw=1, color='grey', linestyle='-')

		plt.subplots_adjust(left=0.3, bottom=0.15, right=0.95, top=0.99, wspace=0, hspace=0)
		x=1

		# fig = plt.figure(figsize=(35, 7), dpi=120, facecolor='w')
		#plt.rcParams.update({'font.size': 7})
		# for subject in np.unique(data.subject):
		# 	axis = fig.add_subplot(int(np.ceil(len(np.unique((data.subject)))/2)), 2, subject + 1)
		# 	subject_model_df = data[(data.subject == subject)]
		# 	sns.barplot(x='model_type', y=criterion, hue='model_struct', data=subject_model_df, ax=axis, )#order=models_order_df(data))
		# 	axis.set_title('Subject:{}'.format(subject+1))
		# 	minn = np.min(data[criterion])
		# 	maxx= np.max(data[criterion])
		# 	delta = 0.1*(maxx-minn)
		# 	axis.set_ylim([minn-delta,maxx+delta])
		# 	labels = axis.get_xticklabels()
		# 	axis.set_ylabel("")
		# 	#axis.set_yticklabels("") if subject % 2 > 0 else 0
		# 	axis.set_xlabel("") if subject <8  else 0
		# 	plt.subplots_adjust(left=0.15, bottom=0.1, right=0.97, top=0.9, wspace=0.2, hspace=0.7)
		#
		# 	params_list = np.round(fitting_utils.string2list(subject_model_df.parameters.tolist()[0]), 3)
		# 	axis.set_title('S{}: {}'.format(subject, params_list))
		# 	despine(axis)

		#fig.subplots_adjust(left=0.02, bottom=0.05, right=0.99, top=0.97, wspace=0.2, hspace=0.1)
	

def average_likelihood_animal(data_file_path, models=None):
	plt.rcParams.update({'font.size': 16})
	df = pd.read_csv(data_file_path)
	data = df[['subject', 'model', 'likelihood', 'day in stage', 'trial', 'stage', 'reward']].copy()
	data = data.groupby(['subject', 'model']).mean().reset_index()
	data = rename_models(data)
	if models:
		data = data[data.model.isin(models)]
	criterion = 'likelihood'
	data = sort_subject_by_criterion(data, criterion)

	plt.figure(figsize=(8, 4), dpi=120, facecolor='w')
	sns.set_palette(triplet_colors(3))
	axis = sns.scatterplot(y='subject', x=criterion, hue='model', alpha=1, data=data,
						   hue_order=models_order_df(data))
	#plt.ylim([0.55, 0.68])

	plt.subplots_adjust(left=0.05, bottom=0.15, right=0.97, top=0.95, wspace=0.2, hspace=0.4)
	despine(axis)

	axis.set_ylabel('Animal')
	axis.set_xlabel('Average Likelihood')

	#handles, labels = axis.get_legend_handles_labels()
	#displabels = [label if i%3==0 else '' for i,label in enumerate(labels)]
	#axis.legend(handles, displabels, loc="upper left", prop={'size': 11},labelspacing=0)
	axis.legend([], [], frameon=False)

	colors = sns.color_palette()
	averages = data.groupby(['model'], sort=False).mean().reset_index()
	averages = averages[['model', 'likelihood']]
	#for ind, (model, likelihood) in enumerate(list(averages.itertuples(index=False, name=None))):
	for ind, model in enumerate(models_order_df(data)):
		likelihood = averages[averages.model==model].likelihood.values[0]
		axis.axvline(x=likelihood, alpha=0.7, lw=2, color=colors[ind])


def average_likelihood_simple(data_file_path, models, pairs=[], criterion='likelihood'):
	plt.rcParams.update({'font.size': 15})
	df = pd.read_csv(data_file_path)
	df = df[['subject', 'model', 'likelihood', 'day in stage', 'trial', 'stage', 'reward']].copy()

	#df = df[df.reward==0]
	df['LL'] = np.log(df.likelihood)
	df['NLL'] = -np.log(df.likelihood)

	df = rename_models(df)

	df=df[df.model.isin(models)]

	df = extract_model_name_type(df)

	# For Stimuli Context dependant figure
	#sns.set_palette("magma", len(models_order_df(df)))

	# For Motivation dependant figure
	if len(models)>5:
		sns.set_palette("Greys", n_colors=3)
	else:
		sns.set_palette("colorblind")

	fig = plt.figure(figsize=(5, 4), dpi=120, facecolor='w')

	if criterion == 'likelihood':
		df = df.groupby(['subject', 'model', 'model_type', 'model_struct']).mean().reset_index()
		args = dict(x="model_type", y=criterion, hue='model_struct', data=df, hue_order=fitting_utils.models_struct_order_df(df))
		axis = sns.barplot(**args, fill=True, errorbar='se')
		minn = 0.32
		maxx = 0.38
		delta = 0.1 * (maxx - minn)
		plt.ylim([minn - delta, maxx + delta])

		one_way_anova(df, criterion, 'model_struct')

		df['trial_id'] = df.apply(lambda  row: str(row.subject) + '_' + str(row.stage) +'_' + str(row['day in stage']) + '_'+str(row.trial), axis=1)
		RM_anova(df, criterion, 'trial_id', 'model_struct')

		if pairs:
			annot = Annotator(axis, pairs, **args)
			annot.configure(test='t-test_paired', text_format='star', loc='inside', line_height=0.01, verbose=2,
							comparisons_correction="bonferroni")
			annot.apply_test().annotate()
		axis.set_ylabel('Average Trial Likelihood')
	else:
		data = df.groupby(['subject', 'model', 'model_type', 'model_struct']).agg(
			{'reward': 'count', 'LL': 'sum', 'likelihood': 'mean'}).reset_index()
		data = data.rename(columns={'reward': 'n'})
		data['k'] = data.apply(lambda row: 3 if row.model in ['SARL','ORL','FRL', 'M(V)-SARL','M(V)-ORL','M(V)-FRL'] else 4, axis=1)
		data['AIC'] = - 2 * data.LL + 2 * data.k
		data['ML'] = data.likelihood / data.n
		data['Geom_avg'] = scipy.stats.mstats.gmean(data.likelihood, nan_policy='omit')

		args = dict(x="model_type", y=criterion, hue='model_struct', data=data)
		axis = sns.barplot(**args, fill=True, errorbar='se', hue_order=fitting_utils.models_struct_order_df(df))

		minn = np.min(data[criterion])
		maxx = np.max(data[criterion])
		axis.set_ylabel(criterion)

	delta = np.max([0.1 * (maxx - minn),0.05])
	plt.ylim([minn - delta, maxx + delta])

	axis.legend([], [], frameon=False)
	handles, labels = axis.get_legend_handles_labels()
	#axis.legend(handles, labels, loc="lower right", prop={'size': 14}, labelspacing=0.2)

	axis.set_xlabel('')
	despine(axis)
	plt.subplots_adjust(left=0.23, bottom=0.1, right=0.97, top=0.98, wspace=0.2, hspace=0.7)


	x=1

#################### Parameters  ###########################

def show_fitting_parameters(data_file_path, models=None):
	params = ['nmr', 'beta', 'alpha', 'alpha_bias']

	df_all = pd.read_csv(data_file_path)
	df = df_all[['subject', 'model', 'parameters', 'likelihood']].copy()

	df = rename_models(df)
	df = unbox_parameters(df, params)

	param_mean = df.groupby(['model']).mean().reset_index()
	param_std = df.groupby(['model']).sem().reset_index()

	params_info = param_mean.merge(param_std, on=['model'])
	params_info = params_info.sort_values(['model'], ascending=False)

	for parameter in params:
		params_info[parameter] = params_info.apply(lambda row: "${:.2} \\pm {:.2}$".format(row[parameter+'_x'], row[parameter+'_y']), axis=1)

	params_info = params_info[['model']+params]
	print(params_info)

	plt.figure()
	ax = sns.scatterplot(data=df, x='alpha', y='alpha_bias', hue='model')
	# # ax.set_xlim([-0.01, 0.1])
	# # ax.set_ylim([-0.01, 0.1])
	# ax = sns.pairplot(hue='model', data=df, diag_kind="hist")
	ax.set(xscale="log", yscale="log")
	ax.legend([], [], frameon=False)
	
def average_nmr_animal(data_file_path, models=None):
	plt.rcParams.update({'font.size': 16})
	df = pd.read_csv(data_file_path)
	data = df[['subject', 'model', 'likelihood', 'day in stage', 'trial', 'stage', 'reward', 'parameters']].copy()

	data = unbox_parameters(data, params=['nmr','beta', 'alpha', 'alpha_bias'])
	data = rename_models(data)
	if models:
		data = data[data.model.isin(models)]
	else:
		data = data[data.model.str.contains('FRL')]
		data = data[~data.model.str.contains('E\(VB\)')]

	criterion = 'nmr'
	data = sort_subject_by_criterion(data, criterion)
	data.subject= data.subject.astype(int)

	fig = plt.figure(figsize=(8, 4), dpi=120, facecolor='w')
	sns.set_palette("tab10", n_colors=len(models_order_df(data)))
	axis = sns.scatterplot(y='subject', x=criterion, hue='model', alpha=1, data=data, s=20,
						  hue_order=models_order_df(data))

	despine(axis)

	axis.set_ylabel('Animal')
	axis.set_xlabel(r'$nmr$')

	axis.legend([], [], frameon=False)
	handles, labels = axis.get_legend_handles_labels()
	# displabels = [label if i%3==0 else '' for i,label in enumerate(labels)]
	axis.legend(handles, labels, loc="upper left", prop={'size': 11},
				labelspacing=0,  fancybox=False, framealpha=0)

	colors = sns.color_palette()
	averages = data.groupby(['model'], sort=False).mean().reset_index()
	averages['likelihood'] = (averages['likelihood'] - averages['likelihood'].min()) / (
				averages['likelihood'].max() - averages['likelihood'].min())
	#averages = averages[['model', criterion]]
	#for ind, (model, likelihood) in enumerate(list(averages.itertuples(index=False, name=None))):
	for ind, model in enumerate(models_order_df(data)):
		likelihood = averages[averages.model==model][criterion].values[0]
		width = (np.exp(averages[averages.model==model].likelihood.values[0]))**2
		axis.axvline(x=likelihood, alpha=0.5, lw=width, color=colors[ind])

	plt.subplots_adjust(left=0.12, bottom=0.15, right=0.97, top=0.95, wspace=0.2, hspace=0.4)

	x=1


def bias_effect_nmr(data_file_path):
	plt.rcParams.update({'font.size': 16})
	df = pd.read_csv(data_file_path)
	data = df[['subject', 'model', 'parameters']].copy()

	data = unbox_parameters(data, params=['nmr', 'beta', 'alpha', 'alpha_bias'])
	data = rename_models(data)
	models = utils.flatten_list([[m, 'B-'+m, 'M(V)-'+m, 'M(VB)-'+m] for m in ['SARL', 'ORL', 'FRL']])

	data = data[data.model.isin(models)]
	data = extract_model_name_type(data)

	data = data[['subject', 'model_type', 'model_struct','nmr', ]].copy()

	data['is_model_biased'] = data.apply(lambda row: 'w/AB' if 'B' in row.model_struct else 'wo/AB', axis=1)
	data['meta_struct'] = data.apply(lambda row: 'M(V), M(VB): '+row.model_type if 'M' in row.model_struct else 'm, B-m: '+row.model_type, axis=1)

	sns.set_palette(triplet_colors(3))
	fig = plt.figure(figsize=(5, 4), dpi=120, facecolor='w')
	# axis = sns.lineplot(x='is_model_biased', y='nmr', hue='meta_struct', data=data,
	# 				alpha=0.8, markers='*', errorbar='se', err_style='bars', sort=False)

	axis = sns.pointplot(x='is_model_biased', y='nmr', hue='meta_struct', data=data,
						 errorbar='se')

	axis.axvline(x=0.5, alpha=0.5,linestyle='--', lw=2, color='gray')
	axis.set_ylim([-1,1])
	despine(axis)

	plt.subplots_adjust(left=0.2, bottom=0.1, right=0.83, top=0.95, wspace=0.2, hspace=0.4)
	axis.legend([], [], frameon=False)
	handles, labels = axis.get_legend_handles_labels()
	fig.legend(handles, labels, loc="upper right", prop={'size': 13},
				labelspacing=0, fancybox=False, framealpha=0.1)

	axis.set_xlabel('')

	x=1
	
#################### Parameters and variables development ###########################

def model_parameters_development(data_file_path, show_per_subject=False):
	plt.rcParams.update({'font.size': 14})
	df_all = pd.read_csv(data_file_path)
	df = df_all[['subject', 'model', 'parameters', 'stage', 'day in stage', 'trial', 'model_variables', 'likelihood']].copy()

	df = df[df['model_variables'].notna()].reset_index()
	df, order, st = index_days(df)
	df = rename_models(df)
	df = filter_days(df)

	# this is needed because there is no good way to order the x-axis in lineplot.
	df.sort_values('ind', axis=0, ascending=True, inplace=True)

	for model in np.unique(df.model): # ['AARL','ACLNet2']:
		df_model = df[df.model == model]
		df, order, st = index_days(df)

		df_model, variables_names= unbox_model_variables(df_model)


		variables_names = [name for name in list(variables_names) if 'none' not in name]
		#variables_names.remove('none')

		fig = plt.figure(figsize=(7, 3), dpi=120, facecolor='w')
		sns.set_palette("colorblind", n_colors=5)
		axis = fig.add_subplot(111)
		for variable_name in variables_names:
			axis = sns.lineplot(x="ind", y=variable_name, data=df_model, errorbar="se", err_style='band', ax=axis, label=variable_name.split('_')[0], marker='o')
		#axis.legend([], [], frameon=False)

		for stage_day in st:
			axis.axvline(x=stage_day - 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2, color='gray')

		axis.axhline(y=0, alpha=0.7, lw=1, color='grey', linestyle='-')

		despine(axis)
		dilute_xticks(axis,1)

		plt.xlabel('Day in Stage. n={}'.format(len(np.unique(df_model.subject))))
		plt.ylabel('Action Bias Norm')

		axis.text(0.01, 0.5, 'Bias to Food Arms', va='center', rotation='vertical')

		plt.title(model)

		#
		# handles, labels = axis.get_legend_handles_labels()
		# plt.legend(handles, labels, loc="upper left", prop={'size': 16}, labelspacing=0.2)
		# plt.subplots_adjust(left=0.12, bottom=0.15, right=0.97, top=0.9, wspace=0.2, hspace=0.4)

		handles, labels = axis.get_legend_handles_labels()
		#axis.legend(handles, ['Water', 'Food'], loc='upper left', prop={'size': 16}, labelspacing=0.4)
		plt.subplots_adjust(left=0.14, bottom=0.2, right=0.99, top=0.9, wspace=0.2, hspace=0.8)

		plt.savefig('fitting/Results/paper_figures/attention_{}'.format(utils.get_timestamp()))

		if show_per_subject:
			plt.rcParams.update({'font.size': 10})
			fig = plt.figure(figsize=(12, 5), dpi=120, facecolor='w')
			animals_ind = np.unique(df_model.subject)
			for i, subject in enumerate(animals_ind):

				df_sub = df_model[df_model.subject == subject]
				axis = fig.add_subplot(int(np.ceil(len(animals_ind)/5)), 5, i+1)

				_, order, st_s = index_days(df_sub)

				for variable_name in variables_names:
					axis = sns.lineplot(x="ind", y=variable_name, data=df_sub, errorbar="se", err_style='band', ax=axis, label=variable_name.split('_')[0])

				for stage_day in st_s:
					axis.axvline(x=stage_day - 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2, color='gray')

				axis.legend([], [], frameon=False)
				despine(axis)
				axis.axhline(y=0, alpha=0.7, lw=1, color='grey', linestyle='-')

				params_list = np.round(fitting_utils.string2list(df_sub.parameters.tolist()[0]),3)
				#axis.set_title('S{}: {}'.format(subject, params_list))

				axis.set_xlabel('Training Day in Stage') if i > len(animals_ind)-3 else axis.set_xlabel('')
				axis.set_ylabel('') #axis.set_ylabel("Attention") if i % 2 == 0 else axis.set_ylabel('')
				plt.tick_params(axis='both', which='major', labelsize=9)

				dilute_xticks(axis,2)

			plt.suptitle(model)
			fig.text(0.01, 0.5, 'Bias to Food Arms', va='center', rotation='vertical')
			plt.subplots_adjust(left=0.06, bottom=0.07, right=0.99, top=0.90, wspace=0.2, hspace=0.8)

			handles, labels = axis.get_legend_handles_labels()
			fig.legend(handles, ['Water Motivation', 'Food Motivation'], loc="upper left", prop={'size': 10}, labelspacing=0.2)

	x = 1
	
	
def model_values_development(data_file_path, rel_models=None, initial_motivation='water'):
	plt.rcParams.update({'font.size': 12})
	df = pd.read_csv(data_file_path)

	df = rename_models(df)
	if rel_models:
		df = df[df.model.isin(rel_models)]

	df = df[df.initial_motivation==initial_motivation]
	df = df[['subject', 'model', 'stage', 'day in stage', 'trial', 'reward', 'model_reward', 'action', 'stimuli_value', 'action_bias']].copy()
	
	df['reward_value'] = df.apply(lambda row: row.reward * (row.action in [3, 4]) if row.stage != 3 else row.reward * (row.action in [1, 2]), axis=1)

	df['delta'] = df.reward_value - (df.stimuli_value + df.action_bias)  # r-Q
	
	model_df = df.groupby(['subject', 'model', 'stage', 'day in stage'], sort='subject').mean().reset_index()


	model_df = filter_days(model_df)
	model_df, order, tr = index_days(model_df)
	# this is needed because there is no good way to order the x-axis in lineplot.
	model_df.sort_values('ind', axis=0, ascending=True, inplace=True)

	colors = sns.color_palette("mako_r", n_colors=2)
	models=models_order_df(model_df)
	fig = plt.figure(figsize=(7, 5), dpi=120, facecolor='w')
	for model_ind, model in enumerate(models):
		axis = fig.add_subplot(len(models), 2, 2*model_ind + 1)
		curr_model_df = model_df[model_df.model==model]
		axis = sns.lineplot(x="ind", y="stimuli_value", #hue="model", hue_order=models_order_df(model_df),
						data=curr_model_df, errorbar="se", err_style='band', color=colors[0], legend=True,label=r'$V(s,a)$')
		axis = sns.lineplot(x="ind", y="delta",  # hue="model", hue_order=models_order_df(model_df),
							data=curr_model_df, errorbar="se", err_style='band', color='grey', legend=True, label=r'$\delta$')
		
		# axis = sns.lineplot(x="ind", y="reward_value",  # hue="model", hue_order=models_order_df(model_df),
		# 					data=curr_model_df, errorbar="se", err_style='band', color='red', legend=True, )
		


		ax2 = fig.add_subplot(len(models), 2, 2*model_ind + 2)
		ax2 = sns.lineplot(x="ind", y="action_bias", #hue="model", hue_order=models_order_df(model_df),,
						data=curr_model_df, errorbar="se", err_style='band', color=colors[1],label=r'$B(a)$')


		for stage_day in tr:
			axis.axvline(x=stage_day - 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2, color='gray')
			ax2.axvline(x=stage_day - 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2, color='gray')

		axis.axhline(y=0, alpha=0.5, lw=1, color='gray')
		ax2.axhline(y=0, alpha=0.5, lw=1, color='gray')

		axis.set(xticklabels=[]) if model_ind<len(models)-1 else 0
		ax2.set(xticklabels=[]) if model_ind < len(models) - 1 else 0

		axis.set_xlabel('')
		ax2.set_xlabel('')
		axis.set_ylabel('')
		ax2.set_ylabel('')
		axis.legend([], [], frameon=False)
		ax2.legend([], [], frameon=False)

		# fig.text(0.01, 0.5, 'V(s,a)', va='center', rotation='vertical', color=colors[0] )
		# fig.text(0.95, 0.5, 'B(a)', va='center', rotation='vertical', color=colors[1])

		# axis.spines['top'].set_visible(False)
		# ax2.spines['top'].set_visible(False)
		despine(axis)
		despine(ax2)

	axis.set_xlabel('Day in Stage')
	ax2.set_xlabel('Day in Stage')

	dilute_xticks(axis,3)
	dilute_xticks(ax2, 3)
	
	handles1, labels1 = axis.get_legend_handles_labels()
	#fig.legend(handles, labels, loc="upper left", prop={'size': 14}, labelspacing=0.2)

	handles2, labels2 = ax2.get_legend_handles_labels()
	fig.legend(handles1+handles2, labels1+labels2, loc="upper center", prop={'size': 10}, labelspacing=0.0)


	plt.subplots_adjust(left=0.08, bottom=0.12, right=0.99, top=0.95, wspace=0.2, hspace=0.2)

	x=1


def bias_variables_in_stage(data_file_path, models=None, initial_motivation='water'):
	"""Showing the action bias estimated by the models for each stage containing
	only rats that particiapted in the entire reported days of the stage."""
	plt.rcParams.update({'font.size': 14})
	df_all = pd.read_csv(data_file_path)
	df = df_all[
		['subject', 'initial_motivation','model', 'parameters', 'stage', 'day in stage', 'trial', 'model_variables', 'likelihood']].copy()

	df = df[df.initial_motivation==initial_motivation]
	
	df = df[df['model_variables'].notna()].reset_index()

	df = rename_models(df)
	df = filter_days(df)
	df, order, st = index_days(df)

	if models:
		df = df[df.model.isin(models)]
	else: 
		models=np.unique(df.model)

	days_per_rat = df[['subject', 'stage', 'day in stage']].drop_duplicates()
	days_per_rat = days_per_rat.groupby(['subject', 'stage'], ).max().reset_index()

	min_days_in_stage = [5, 2, 3, 2, 2]
	#min_days_in_stage = [1, 1, 3, 1, 1]
	sns.set_palette("colorblind", n_colors=2)
	fig = plt.figure(figsize=(7, 5), dpi=120, facecolor='w')

	for model_ind, model in enumerate(models):
		model_df = df[df.model==model]

		data_for_model = pd.DataFrame()
		for stg_ind, stage in enumerate(stages):
			#find the rates that maintained at least the number of days: 5,2,3,2,2

			min_days = min_days_in_stage[stg_ind]
			relevant_rats = days_per_rat[(days_per_rat.stage == stg_ind + 1) & (days_per_rat['day in stage'] >= min_days)]
			relevant_rats = relevant_rats.subject

			#read data of relevant rats in the stage
			df_relevant_rats = model_df[(model_df.subject.isin(relevant_rats)) & (model_df.stage == stg_ind + 1)
										& (model_df['day in stage'] <= min_days_in_stage[stg_ind])]


			df_relevant_rats, variable_names = unbox_model_variables(df_relevant_rats)
			variable_names = [name for name in list(variable_names) if 'none' not in name] #['water'] #

			data_for_model = pd.concat([data_for_model,df_relevant_rats], axis=0)
		

		# averages_per_day = data_for_model.groupby(['subject','ind','stage', 'day in stage'], sort=False).mean().reset_index()
		# averages_per_day, order, st = index_days(averages_per_day)

		data_for_model, order, st = index_days(data_for_model)

		axis = fig.add_subplot(len(models), 1, model_ind + 1)
		for variable_name in variable_names:
			axis = sns.lineplot(x="ind", y=variable_name, data=data_for_model, errorbar="se",
								err_style='band', ax=axis, label=variable_name.split('_')[0])

		for stage_day in st:
			axis.axvline(x=stage_day - 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2, color='gray')

		axis.axhline(y=0, alpha=0.5, lw=1, color='gray')

		axis.set_xlabel('')
		axis.set_ylabel(model)
		axis.legend([], [], frameon=False)
		despine(axis)

		axis.set_xticklabels([]) if model_ind<len(models)-1 else 0
	axis.set_xlabel('Training day in stage')
	dilute_xticks(axis,2)
	handles, labels = axis.get_legend_handles_labels()
	#fig.legend(handles, ['Water Motivation','Food Motivation'], loc="upper center", prop={'size': 11}, labelspacing=0.2)

	fig.text(0.01, 0.5, 'Bias to Food Arms', va='center', rotation='vertical')

	plt.subplots_adjust(left=0.17, bottom=0.12, right=0.99, top=0.95, wspace=0.2, hspace=0.2)

	x=1

###################### Experimental Section ##########################

def daily_success_vs_likelihood(data_file_path, models=None):
	plt.rcParams.update({'font.size': 16})
	df = pd.read_csv(data_file_path)
	df['model_rat_action'] = df.model_action == df.action
	df = df[df.reward==0]

	df = rename_models(df)

	data = df[['subject', 'model', 'likelihood', 'day in stage', 'trial', 'stage', 'reward', 'model_reward', 'model_rat_action']].copy()
	data = data.groupby(['subject', 'model','stage', 'day in stage'], sort=False).mean().reset_index()



	if models is not None:
		data = data[data.model.isin(models)]

	axis = sns.scatterplot(data=data, x='likelihood', y='model_rat_action', hue='model', s=7)

	axis.set_xlim([0,1])
	axis.set_ylim([0, 1])

	x=1








