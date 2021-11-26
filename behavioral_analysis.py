from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

def plot_days_per_stage(reports_pg, reports_dqn):
	stages = list(range(0, 5))
	repetitions = len(reports_pg)
	days_per_stage_pg = []
	days_per_stage_dqn = []
	for experiment_report_df_pg in reports_pg:
		c = Counter(list(experiment_report_df_pg['Stage']))
		days_per_stage_pg.append([c[i] for i in stages])

	for experiment_report_df_dqn in reports_dqn:
		c = Counter(list(experiment_report_df_dqn['Stage']))
		days_per_stage_dqn.append([c[i] for i in stages])

	days_per_stage_pg = np.stack(days_per_stage_pg)
	days_per_stage_dqn = np.stack(days_per_stage_dqn)

	width = 0.25
	X = np.array(stages)
	fig, ax = plt.subplots(figsize=(10, 6))
	ax.bar(stages, np.mean(days_per_stage_pg, axis=0), yerr=np.std(days_per_stage_pg, axis=0), color='b', width=width,
		   label='PG', capsize=2)
	ax.bar(stages + width, np.mean(days_per_stage_dqn, axis=0), yerr=np.std(days_per_stage_dqn, axis=0),
		   color='g', width=width, label='DQN', capsize=2)
	plt.title("Days Per stage - PG vs DQN. #reps={}".format(repetitions))
	plt.legend()
	plt.show()


def days_to_consider_in_each_stage(reports, q=75):
	stages = list(range(0, 5))
	days_per_stage = []
	for experiment_report_df in reports:
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
	signals = ['Reward', 'Correct', 'WaterPreference', 'WaterCorrect', 'FoodCorrect']

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
	X = np.array(list(range(0,stage_indices[-1])))+1

	formats = ['g+-', 'y-', '^-', 'bo-', 'ro-']
	for i,signal in enumerate(signals):
		ax = plt.errorbar(X, np.nanmean(results_dict[signal], axis=0), yerr=np.nanstd(results_dict[signal], axis=0), fmt=formats[i], label=signal, alpha=0.6)

	for stage in stage_indices[1:]:
		plt.axvline(x=stage + 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2)

	plt.xlabel('Days')
	plt.ylabel('Percent')

	plt.title("Behavioral Stats of {} individuals. brain{}. #params={}".format(
		len(reports), reports[0]._metadata['brain'], reports[0]._metadata['brain_params']))
	plt.legend()

	# self._line_water_preference, = self._axes_graph.plot([], [], '^-', label='Water PI', markersize=3, alpha=0.4)
	# self._line_water_correct, = self._axes_graph.plot([], [], 'bo-', label='Water Correct', markersize=3)
	# self._line_food_correct, = self._axes_graph.plot([], [], 'ro-', label='Food Correct', markersize=3)
