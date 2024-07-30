import os.path

import pandas as pd
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator

import utils
from fitting import fitting_utils
from fitting.MazeResultsBehavioural import model_parameters_development, \
	model_parameters_development_correct_vs_incorrect
from fitting.MazeResultsBehaviouralLC import index_days, calculate_goal_choice, dilute_xticks, despine
from fitting.fitting_utils import stable_unique, rename_models, models_order_df
import seaborn as sns
from environment import PlusMazeOneHotCues, PlusMazeOneHotCues2ActiveDoors
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'font.size': 14})


stages = PlusMazeOneHotCues2ActiveDoors.stage_names

figures_folder = '/Users/georgekour/repositories/plus-maze-simulator/fitting/Results/figures'


def show_days_to_criterion(simulation_df, stages=stages):
	df = pd.read_csv(simulation_df)
	df = df[['subject', 'model', 'stage', 'day in stage', 'trial', 'reward']].copy()

	df = rename_models(df)
	relevant_models = models_order_df(df)
	df = df[df.model.isin(relevant_models)]
	df['day in stage']=(df['day in stage']+1)
	sns.set_palette('OrRd', n_colors=len(stages))
	df = df.groupby(['subject', 'model', 'stage'], sort=False).agg({'day in stage': 'max'}).reset_index()
	fig = plt.figure(figsize=(6.5, 3.5))
	g1 = sns.barplot(x='model', y='day in stage', hue='stage', hue_order=list(range(1, len(stages)+1)),
					 data=df, errorbar='se', errwidth=1, capsize=.05, order=relevant_models)

	# cols = ['green' if stage > 1 else 'red' for stage in df.stage]
	# stage_names = [stages[stage-1] for stage in df.stage]
	# g1 = sns.barplot(x=stage_names, y='day in stage', palette=cols, data=df, errorbar='se', errwidth=1, capsize=.05)

	g1.set(xlabel='', ylabel='Days Until Criterion')
	handles, labels = g1.get_legend_handles_labels()
	g1.legend(handles, stages, loc='upper left', prop={'size': 12}, labelspacing=0.2)

	despine(g1)

	plt.savefig(os.path.join(figures_folder,'simulation_criterion_days_{}.pdf'.format(utils.get_timestamp())))


def water_preference_index_model(data_file_path):
	sns.set_style("ticks",{'axes.grid' : True})
	plt.rcParams.update({'font.size': 16})

	df = pd.read_csv(data_file_path)

	df['day in stage'] += 1
	#df = df[df.model==df.model[0]]

	df['initial_motivation'] = 'water'
	dd = calculate_goal_choice(df)

	dd = rename_models(dd)
	dd, order, tr = index_days(dd)

	fig = plt.figure(figsize=(10, 5))
	axis = sns.pointplot(x="ind", y="gc", hue="model", data=dd, errorbar="se", join=False, order=order,
						 hue_order=models_order_df(dd),
						 capsize=.2, palette=sns.color_palette("colorblind"), scale=0.7, dodge=0.2, linestyles='--')

	for stage_day in tr:
		axis.axvline(x=stage_day - 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2, color='gray')
	axis.axhline(y=0, alpha=0.5, lw=2, color='gray', linestyle='--')

	axis.set_ylim([-1, 1])
	despine(axis)
	dilute_xticks(axis,2)
	axis.set_ylabel('Water Preference Index')
	axis.set_xlabel('Training Day in Stage')

	axis.legend([], [], frameon=False)
	handles, labels = axis.get_legend_handles_labels()
	axis.legend(handles, labels, loc="lower right", prop={'size': 16},
			   labelspacing=0.3)

	plt.grid(color='whitesmoke', linestyle='-', linewidth=20, which='major', axis='x')
	axis.yaxis.grid(False)

	plt.subplots_adjust(left=0.1, bottom=0.15, right=0.99, top=0.95, wspace=0.1, hspace=0.4)


	x=1


def num_days_till_criterion_increasing_ID(file_path):
	all_simulation_df = pd.read_csv(file_path)

	all_simulation_df['day in stage']=all_simulation_df['day in stage']+1
	df_days_in_stage = all_simulation_df.groupby(['subject', 'env_setup', 'stage'])['day in stage'].max().reset_index()
	df_days_in_stage['num_odor_stages'] = df_days_in_stage['env_setup'].apply(lambda x: x.count('Odor'))

	max_days_in_LED = df_days_in_stage[df_days_in_stage['stage']==(df_days_in_stage['num_odor_stages']+1)]
	max_days_in_last_odor_stage = df_days_in_stage[(df_days_in_stage['env_setup'] == 'Odor5,Odor4,Odor3,Odor2,Odor1,LED') &(df_days_in_stage['stage']<=df_days_in_stage['num_odor_stages'])]

	#(df_days_in_stage['env_setup'] == 'Odor5,Odor4,Odor3,Odor2,Odor1,LED') &

	fig = plt.figure(figsize=(5, 3), dpi=120, facecolor='w')

	g1 = sns.lineplot(data=max_days_in_LED, x='num_odor_stages', y='day in stage', marker='o', errorbar='se', color='black')

	g1.set_xlabel('Number of Odor Stages')
	g1.set_ylabel('Days till criterion in LED Stage (EDS)', color='black')

	#plt.title('Average Days vs. Number of Odor Stages')

	# Create a second y-axis
	ax2 = g1.twinx()

	# Plot the second line on the right axis
	sns.lineplot(data=max_days_in_last_odor_stage, x='stage', y='day in stage', marker='o',
					  errorbar='se', color='blue', ax=ax2)
	ax2.set_ylabel('Days till criterion in Last Odor Stage (IDS)', color='blue')
	ax2.tick_params(axis='y', labelcolor='blue')

	# Set the y-axis limits for both axes
	# g1.set_ylim(0, 9)
	# ax2.set_ylim(0, 9)

	plt.show()
	# despine(g1)
	g1.spines['top'].set_visible(False)
	plt.savefig(os.path.join(figures_folder,'num_days_till_criterion_increasing_ID_{}.pdf'.format(utils.get_timestamp())))


def num_days_till_criterion_odor_stages(file_path):
	all_simulation_df = pd.read_csv(file_path)

	many_odor_df = all_simulation_df[
		(all_simulation_df["env_setup"] == "Odor5,Odor4,Odor3,Odor2,Odor1,LED") & (all_simulation_df["stage"]!=all_simulation_df["stage"].max())]
	max_days = many_odor_df.groupby(['subject', 'stage'])['day in stage'].max().reset_index()

	# Create the plot
	plt.figure(figsize=(10, 6))
	g1 = sns.barplot(data=max_days, x='stage', y='day in stage', errorbar='se')
	plt.xlabel('Odor Stage')
	plt.ylabel('Days to Criterion')
	plt.title('Days to Criterion vs. Odor Stage')

	args = dict(x="stage", y='day in stage')
	pairs = [(4, 5), (3, 4), (3, 5), (2, 4), (2, 5), (2, 3), (1, 2), (1, 3), (1, 4), (1, 5)]

	annot = Annotator(g1, pairs, **args, data=max_days)
	annot.configure(test='t-test_paired', text_format='star', loc='inside', verbose=2,
					comparisons_correction="Bonferroni")

	annot.apply_test().annotate()

	max_days = max_days.rename(columns={'day in stage': 'dayinstage'})
	fitting_utils.one_way_anova(max_days,target='dayinstage',c="stage")
	# despine(g1)
	plt.savefig(os.path.join(figures_folder,'num_days_till_criterion_odor_stages{}.pdf'.format(utils.get_timestamp())))

	plt.show()

if __name__ == '__main__':

	#show_days_to_criterion('/Users/georgekour/repositories/plus-maze-simulator/fitting/Results/Rats-Results/reported_results_dimensional_shifting/simulation_days_50_100TPD.csv')
	show_days_to_criterion('/Users/georgekour/repositories/plus-maze-simulator/fitting/Results/Rats-Results/reported_results_dimensional_shifting/simulation_10_2023_09_OOC.csv')
	#show_days_to_criterion('fitting/Results/simulations_results/simulation_ORL_0nmr.csv')
	#goal_choice_index_model('/fitting/Results/simulations_results/simulation_FRL_0nmr.csv')
	#water_preference_index_model('fitting/Results/Rats-Results/reported_results_motivation_shifting/simulation_FRL_0nmr.csv')
	#num_days_till_criterion_increasing_ID('/Users/georgekour/repositories/plus-maze-simulator/fitting/Results/Rats-Results/reported_results_dimensional_shifting/increasing_ID_simulation_20.csv')
	# num_days_till_criterion_increasing_ID(
	# 	"/fitting/Results/Rats-Results/reported_results_dimensional_shifting/increasing_ID_50_100TPD.csv")
	# #num_days_till_criterion_odor_stages("/Users/georgekour/repositories/plus-maze-simulator/fitting/Results/Rats-Results/reported_results_dimensional_shifting/increasing_ID_simulation_20.csv")

	#show_days_to_criterion('/Users/georgekour/repositories/plus-maze-simulator/fitting/Results/Rats-Results/reported_results_dimensional_shifting/ED_shift_50_100TPD.csv', stages=['Odor', 'LED1', 'LED2', 'LED3' ])
	ED_shift_analysis = '/Users/georgekour/repositories/plus-maze-simulator/fitting/Results/Rats-Results/reported_results_dimensional_shifting/ED_shift_20_100TPD.csv'
	model_parameters_development(ED_shift_analysis, reward_dependant_trials=1)
	#model_parameters_development_correct_vs_incorrect(ED_shift_analysis)
	x=1