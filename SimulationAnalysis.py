import pandas as pd
import matplotlib.pyplot as plt

from fitting.fitting_utils import stable_unique, rename_models
import seaborn as sns

def despine(axis):
	axis.spines['top'].set_visible(False)
	axis.spines['right'].set_visible(False)


stages = ['ODOR1', 'ODOR2', 'LED']
plt.rcParams.update({'font.size': 16})


def show_days_to_criterion(simulation_df):
	df = pd.read_csv(simulation_df)
	df = df[['subject', 'model', 'stage', 'day in stage', 'trial', 'reward']].copy()

	df = rename_models(df)

	sns.set_palette('OrRd', n_colors=3)
	df = df.groupby(['subject', 'model', 'stage'], sort=False).agg({'day in stage': 'max'}).reset_index()
	fig = plt.figure(figsize=(6.5, 3.5))
	g1 = sns.barplot(x='model', y='day in stage', hue='stage', hue_order=[1,2,3],
					 data=df, errorbar='se', errwidth=1, capsize=.05)

	g1.set(xlabel='', ylabel='Days Until Criterion')
	handles, labels = g1.get_legend_handles_labels()
	g1.legend(handles, stages, loc='upper left', prop={'size': 12}, labelspacing=0.2)

	despine(g1)


if __name__ == '__main__':
	show_days_to_criterion('/Users/gkour/repositories/plusmaze/fitting/Results/simulations_results/simulation_20_2023_01_12_02_35.csv')



