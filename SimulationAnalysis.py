import pandas as pd
import matplotlib.pyplot as plt

from fitting.MazeResultsBehaviouralLC import index_days, calculate_goal_choice, dilute_xticks
from fitting.fitting_utils import stable_unique, rename_models, models_order_df
import seaborn as sns
from environment import PlusMazeOneHotCues, PlusMazeOneHotCues2ActiveDoors

def despine(axis):
	axis.spines['top'].set_visible(False)
	axis.spines['right'].set_visible(False)


stages = PlusMazeOneHotCues2ActiveDoors.stage_names
stages = PlusMazeOneHotCues.stage_names

plt.rcParams.update({'font.size': 10})

def show_days_to_criterion(simulation_df):
	df = pd.read_csv(simulation_df)
	df = df[['subject', 'model', 'stage', 'day in stage', 'trial', 'reward']].copy()

	df = rename_models(df)

	sns.set_palette('OrRd', n_colors=len(stages))
	df = df.groupby(['subject', 'model', 'stage'], sort=False).agg({'day in stage': 'max'}).reset_index()
	fig = plt.figure(figsize=(6.5, 3.5))
	g1 = sns.barplot(x='model', y='day in stage', hue='stage', hue_order=list(range(1, len(stages)+1)),
					 data=df, errorbar='se', errwidth=1, capsize=.05)

	g1.set(xlabel='', ylabel='Days Until Criterion')
	handles, labels = g1.get_legend_handles_labels()
	g1.legend(handles, stages, loc='upper left', prop={'size': 12}, labelspacing=0.2)

	despine(g1)


def goal_choice_index_model(data_file_path):
	plt.rcParams.update({'font.size': 16})
	df = pd.read_csv(data_file_path)

	df['day in stage'] += 1
	#df = df[df.model==df.model[0]]

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
	axis.set_ylabel('Goal Choice')
	axis.set_xlabel('Training Day in Stage')
	plt.subplots_adjust(left=0.08, bottom=0.15, right=0.99, top=0.98, wspace=0.1, hspace=0.4)

	axis.legend([], [], frameon=False)
	handles, labels = axis.get_legend_handles_labels()
	axis.legend(handles, labels, loc="lower right", prop={'size': 16},
			   labelspacing=0.3)


	x=1




if __name__ == '__main__':

	#show_days_to_criterion('/Users/gkour/repositories/plusmaze/fitting/Results/simulations_results/simulation_20_2023_01_13_13_19.csv')
	#show_days_to_criterion('/Users/gkour/repositories/plusmaze/fitting/Results/simulations_results/simulation_10_2023_01_25_18_52.csv')
	goal_choice_index_model('/Users/gkour/repositories/plusmaze/fitting/Results/simulations_results/simulation_10_2023_01_25_18_52.csv')


