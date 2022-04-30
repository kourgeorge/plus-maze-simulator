__author__ = 'gkour'

from collections import Counter
import config
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
import time


def plot_days_per_stage(brains_repetitions_reports):
	stages = list(range(0, 5))
	width = 0.7/len(brains_repetitions_reports)
	fig, ax = plt.subplots(figsize=(10, 6))
	brain_names = []
	for i, brain_reports in enumerate(brains_repetitions_reports):
		repetitions = len(brain_reports)
		days_per_stage_pg = []
		brain_names+=[brain_reports[0]._metadata['brain']]
		for experiment_report_df_pg in brain_reports:
			c = Counter(list(experiment_report_df_pg['Stage']))
			days_per_stage_pg.append([c[i] for i in stages])

		days_per_stage_pg = np.stack(days_per_stage_pg)

		ax.bar(np.array(stages) + width*i, np.mean(days_per_stage_pg, axis=0), yerr=sem(days_per_stage_pg, axis=0, nan_policy='omit'),
			width=width, label="{}:{}".format(brain_reports[0]._metadata['brain'],brain_reports[0]._metadata['brain_params']), capsize=2)

	plt.xticks(np.array(stages) + width/4*len(brains_repetitions_reports),config.stage_names, rotation=0, fontsize='10', horizontalalignment='center')

	plt.title("Days Per stage. #reps={}".format(repetitions))
	plt.legend()

	plt.savefig('Results/days_in_stage_-{}'.format(time.strftime("%Y%m%d-%H%M")))


def days_to_consider_in_each_stage(subject_reports, q=75):
	stages = list(range(0, 5))
	days_per_stage = []
	for experiment_report_df in subject_reports:
		c = Counter(list(experiment_report_df['Stage']))
		days_per_stage.append([c[i] for i in stages])

	days_per_stage = np.array(days_per_stage)
	considered_days_per_stage = [None] * len(stages)
	for stage in stages:
		considered_days_per_stage[stage] = np.int(np.percentile(a=days_per_stage[:, stage], q=q))

	return considered_days_per_stage


def plot_behavior_results(reports):
	stages = list(range(0, 5))
	days_each_stage = days_to_consider_in_each_stage(reports)
	b_signals = ['Correct', 'Reward', 'WaterPreference', 'WaterCorrect', 'FoodCorrect']
	n_signals = ['AffineDim','ControllerDim']

	signals = b_signals+n_signals
	# days_each_stage_sum = np.cumsum(days_each_stage)
	results_dict = {}
	for signal in signals:
		results_dict[signal] = np.ndarray(shape=[len(reports), sum(days_each_stage)])
	stage_indices = np.insert(np.cumsum(days_each_stage),0,0)
	for i, report in enumerate(reports):
		for stage in stages:
			stage_rows_df = report.loc[report['Stage'] == stage]
			days_in_stage_to_consider = np.min([len(stage_rows_df), days_each_stage[stage]])

			for signal in signals:
				rew_ser = list(stage_rows_df[:days_in_stage_to_consider][signal])
				rew_ser = rew_ser + [np.nan] * (days_each_stage[stage] - len(rew_ser))
				results_dict[signal][i, stage_indices[stage]:stage_indices[stage+1]] = rew_ser


	fig = plt.figure(figsize=(9, 5), dpi=120, facecolor='w')
	axes_behavioral_graph = fig.add_subplot(211)
	axes_neural_graph = fig.add_subplot(212)

	X = np.array(list(range(0,stage_indices[-1])))+1

	formats = ['g+-', 'y-', '^-', 'bo-', 'ro-']
	for i,signal in enumerate(b_signals):
		ax = axes_behavioral_graph.errorbar(X, np.nanmean(results_dict[signal], axis=0), yerr=sem(results_dict[signal], axis=0, nan_policy='omit'), fmt=formats[i], label=signal, alpha=0.6)

	formats = ['m^-', 'g^-']
	for i,signal in enumerate(n_signals):
		ax = axes_neural_graph.errorbar(X, np.nanmean(results_dict[signal], axis=0), yerr=sem(results_dict[signal], axis=0, nan_policy='omit'), fmt=formats[i], label=signal, alpha=0.6)

	for stage in stage_indices[1:]:
		axes_behavioral_graph.axvline(x=stage + 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2)
		axes_neural_graph.axvline(x=stage + 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2)

	plt.xlabel('Days')
	plt.ylabel('Percent')

	fig.suptitle("Behavioral Stats of {} individuals. brain:{}. #params:{}.".format(
		len(reports), reports[0]._metadata['brain'], reports[0]._metadata['brain_params']))

	axes_behavioral_graph.legend()
	axes_neural_graph.legend()
	axes_behavioral_graph.set_ylim(0, 1)
	axes_neural_graph.set_ylim(0, 8)

	plt.savefig('Results/Behavioural_stats_{}-{}'.format(reports[0]._metadata['brain'],time.strftime("%Y%m%d-%H%M")))
