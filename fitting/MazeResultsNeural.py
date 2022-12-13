
import numpy as np
import pandas as pd
import fitting_utils
import fitting_config
import matplotlib.pyplot as plt
from environment import PlusMazeOneHotCues2ActiveDoors, CueType

from fitting.PlusMazeExperimentFitting import PlusMazeExperimentFitting
from learners.networklearners import DQNAtt
from learners.tabularlearners import MALearner
from motivatedagent import MotivatedAgent
from rewardtype import RewardType
import config

np.set_printoptions(suppress=True)


def name_to_brain(model_name):
	for brain in fitting_config.maze_models:
		if fitting_utils.brain_name(brain[0]) in model_name:
			return brain[0]
	return None


def neural_display_timeline(data_file_path):
	experiment_data = pd.read_csv(data_file_path, index_col=0)

	subject_models_data = pd.DataFrame()
	for model_name in np.unique(experiment_data.model):
		fig = plt.figure(figsize=(35, 7), dpi=120, facecolor='w')
		plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.90, wspace=0.15, hspace=0.4)
		fig.suptitle(model_name)
		for i, subject in enumerate(np.unique(experiment_data.subject)):
			axis = fig.add_subplot(3, 3, subject + 1)
			print(model_name)
			model_data = experiment_data[(experiment_data.subject == subject) & (experiment_data.model == model_name)]
			model_data.reset_index(inplace=True, drop=True)
			params = list(model_data['parameters'])[0]
			env = PlusMazeOneHotCues2ActiveDoors(relevant_cue=CueType.ODOR, stimuli_encoding=10)

			brain, learner, model = name_to_brain(model_name)
			model_instance = model(env.stimuli_encoding_size(), 2, env.num_actions())

			if issubclass(learner, MALearner) or issubclass(learner, DQNAtt):
				(beta, lr, attention_lr) = params= fitting_utils.string2list(params)
				learner_instance = learner(model_instance, learning_rate=lr, alpha_phi=attention_lr)
			else:
				(beta, lr) = params = fitting_utils.string2list(params)
				learner_instance = learner(model_instance, learning_rate=lr)

			agent = MotivatedAgent(brain(learner_instance),
								   motivation=RewardType.WATER, motivated_reward_value=config.MOTIVATED_REWARD,
								   non_motivated_reward_value=config.NON_MOTIVATED_REWARD)
			fitting_utils.blockPrint()
			stats, fitting_info = PlusMazeExperimentFitting(env, agent, model_data, dashboard=False)
			fitting_utils.enablePrint()

			model_dev = np.stack([np.hstack(day_report.brain.get_model().get_model_metrics().values()) for day_report in stats.reports])
			dict = {'model': model_name,
				  'subject': subject,
				  'model_change': model_dev}

			subject_models_data = subject_models_data.append(dict, ignore_index=True)
			axis.plot(model_dev, label=list(stats.reports[0].brain.get_model().get_model_metrics().keys()))
			axis.set_title('Subject: {}. Params:{}, lik:{}'.format(subject + 1, np.round(params, 4),
																   np.round(np.mean(np.mean(
																	   model_data.groupby('stage').mean().likelihood)),
																			2)))
			days_level_data = model_data.groupby(['stage', 'day in stage']).mean().reset_index()
			for stage_day in fitting_utils.get_stage_transition_days(days_level_data):
				axis.axvline(x=stage_day - 0.5, alpha=0.5, dashes=(5, 2, 1, 2), lw=2)

		handles, labels = axis.get_legend_handles_labels()
		fig.legend(handles, labels, loc=(0.01, 0.87), prop={'size': 10}, labelspacing=0.3)


		plt.savefig('fitting/Results/figures/neural/neural_{}_{}.jpg'.format(model_name.split('.')[0],fitting_utils.get_timestamp()))
		plt.show()


if __name__ == '__main__':

	file_path = '/Users/gkour/repositories/plusmaze/fitting/Results/Rats-Results/fitting_results_2022_09_23_19_29_35.csv'
	neural_display_timeline(file_path)

