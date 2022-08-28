__author__ = 'gkour'

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
import time
from PlusMazeExperiment import stage_names

import utils


def plot_days_per_stage(all_brains_types_stats, file_path):
	stages = list(range(len(stage_names)))
	width = 0.7/len(all_brains_types_stats)
	fig, ax = plt.subplots(figsize=(10, 6))
	brain_names = []
	stats_all_brains = {}
	for i, brain_type_stats in enumerate(all_brains_types_stats):
		repetitions = len(brain_type_stats)
		days_per_stage_brain_type = []
		brain_names+=[brain_type_stats[0].metadata['brain']]
		for experiment_stats in brain_type_stats:
			c = Counter(list(experiment_stats.epoch_stats_df['Stage']))
			days_per_stage_brain_type.append([c[i] for i in stages])

		days_per_stage_brain_type = np.stack(days_per_stage_brain_type)

		ax.bar(np.array(stages) + width*i, np.mean(days_per_stage_brain_type, axis=0), yerr=sem(days_per_stage_brain_type, axis=0, nan_policy='omit'),
			width=width, label="{}:{}({})".format(brain_type_stats[0].metadata['brain'],
												 brain_type_stats[0].metadata['network'],
												 brain_type_stats[0].metadata['brain_params']), capsize=2)
		stats_all_brains[i] = days_per_stage_brain_type

	plt.xticks(np.array(stages) + width / 2 * len(all_brains_types_stats), stage_names, rotation=0, fontsize='10', horizontalalignment='center')

	plt.title("Days Per stage. #reps={}".format(repetitions))
	plt.legend()

	plt.savefig(file_path)


def days_to_consider_in_each_stage(subject_reports, q=75):
	stages = list(range(len(stage_names)))
	days_per_stage = []
	for experiment_report_df in subject_reports:
		c = Counter(list(experiment_report_df.epoch_stats_df['Stage']))
		days_per_stage.append([c[i] for i in stages])

	days_per_stage = np.array(days_per_stage)
	considered_days_per_stage = [None] * len(stages)
	for stage in stages:
		considered_days_per_stage[stage] = np.int(np.percentile(a=days_per_stage[:, stage], q=q))

	return considered_days_per_stage


def plot_behavior_results(brain_type_stats, dirname=None):
	stages = list(range(len(stage_names)))
	days_each_stage = days_to_consider_in_each_stage(brain_type_stats)
	b_signals = ['Correct', 'Reward', 'WaterPreference', 'WaterCorrect', 'FoodCorrect', 'Likelihood']
	b_signals = ['Correct', 'CorrectNetwork', 'Likelihood']
	n_signals = list(brain_type_stats[0].reports[0].brain.get_network().get_network_metrics().keys())+\
				list(brain_type_stats[0].reports[0].brain.get_network().network_diff(brain_type_stats[0].reports[0].brain.get_network()).keys())

	results_dict = {}
	for signal in b_signals+n_signals:
		results_dict[signal] = np.ndarray(shape=[len(brain_type_stats), sum(days_each_stage)])

	stage_indices = np.insert(np.cumsum(days_each_stage),0,0)
	for i, stat in enumerate(brain_type_stats):
		for stage in stages:
			stage_rows_df = stat.epoch_stats_df.loc[stat.epoch_stats_df['Stage'] == stage]
			days_in_stage_to_consider = np.min([len(stage_rows_df), days_each_stage[stage]])

			for signal in b_signals+n_signals:
				rew_ser = list(stage_rows_df[:days_in_stage_to_consider][signal])
				rew_ser = rew_ser + [np.nan] * (days_each_stage[stage] - len(rew_ser))
				results_dict[signal][i, stage_indices[stage]:stage_indices[stage+1]] = rew_ser

	fig = plt.figure(figsize=(9, 5), dpi=120, facecolor='w')
	axes_behavioral_graph = fig.add_subplot(211)
	axes_neural_graph = fig.add_subplot(212)

	X = np.array(list(range(0,stage_indices[-1])))+1

	formats = ['g+-', 'y-', '^-', 'bo-', 'ro-']
	for signal in b_signals:
		ax = axes_behavioral_graph.errorbar(X, np.nanmean(results_dict[signal], axis=0),
											yerr=sem(results_dict[signal], axis=0, nan_policy='omit'), fmt='o-',
											color=utils.colorify(signal), label=signal, alpha=0.6, markersize=2)
		if signal=='CorrectNetwork':
			axes_behavioral_graph = axes_behavioral_graph.twinx()

	for n_sub_signal in n_signals:
		ax = axes_neural_graph.errorbar(X, np.nanmean(results_dict[n_sub_signal], axis=0),
										yerr=sem(results_dict[n_sub_signal], axis=0, nan_policy='omit'),
										color=utils.colorify(n_sub_signal), fmt='^-', label=n_sub_signal, alpha=0.6, markersize=2)

	for stage in stage_indices[1:]:
		axes_behavioral_graph.axvline(x=stage + 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2)
		axes_neural_graph.axvline(x=stage + 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2)

	plt.xlabel('Days')
	#plt.ylabel('Percent')

	fig.suptitle("Stats of {} individuals.\nbrain:{}. network:{}({})".format(
		len(brain_type_stats),
		brain_type_stats[0].metadata['brain'], brain_type_stats[0].metadata['network'],
		brain_type_stats[0].metadata['brain_params']
		),fontsize=8)

	axes_behavioral_graph.legend(prop={'size': 7})
	axes_neural_graph.legend(prop={'size': 7})
	#axes_behavioral_graph.set_ylim(0, 1)
	#axes_neural_graph.set_ylim(0, 0.75)
	#axes_neural_graph.set_yscale('log')

	if dirname is not None:
		plt.savefig('Results/{}/Stats_{}-{}-{}'.format(dirname, brain_type_stats[0].metadata['brain'],brain_type_stats[0].metadata['network'], time.strftime("%Y%m%d-%H%M")))
