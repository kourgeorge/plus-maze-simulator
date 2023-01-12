__author__ = 'gkour'

from brains.lateoutcomeevaluationbrain import LateOutcomeEvaluationBrain
from motivatedagent import MotivatedAgent
from environment import PlusMazeOneHotCues2ActiveDoors, CueType, PlusMazeOneHotCues

import os
import config
from models.networkmodels import *
from learners.networklearners import *
from learners.tabularlearners import *
from brains.fixeddoorattentionbrain import FixedDoorAttentionBrain
from brains.motivationdependantbrain import MotivationDependantBrain
from PlusMazeExperiment import PlusMazeExperiment, ExperimentStatus
from behavioral_analysis import plot_days_per_stage, plot_behavior_results
from brains.consolidationbrain import ConsolidationBrain
from rewardtype import RewardType
from brains.tdbrain import TDBrain

brains = [#(TDBrain, QLearner, QTable),
        #(TDBrain, OptionsLearner, OptionsTable),
        #(TDBrain, IAAluisiLearner, ACFTable),
        #(TDBrain, IALearner, ACFTable),
        (TDBrain, MALearner, ACFTable),
         #(TDBrain, MALearnerSimple, ACFTable),
          # (ConsolidationBrain, DQN, UniformAttentionNetwork),
          #(ConsolidationBrain, DQN, AttentionAtChoiceAndLearningNetwork),
          # (ConsolidationBrain, DQN, FullyConnectedNetwork),
          #(ConsolidationBrain, DQN, FC2LayersNet),
          # (ConsolidationBrain, DQN, EfficientNetwork),
          # (FixedDoorAttentionBrain, DQN, EfficientNetwork),
          # (MotivationDependantBrain, DQN, SeparateMotivationAreasNetwork),
          # (MotivationDependantBrain, DQN, SeparateMotivationAreasFCNetwork),
          # (LateOutcomeEvaluationBrain, DQN, SeparateMotivationAreasNetwork)
          ]


def run_simulation(env):
    repetitions = 20

    all_experiment_data = pd.DataFrame()
    brains_reports = []
    for agent_spec in brains:
        completed_experiments = 0
        aborted_experiments = 0
        brain_repetition_reports = [None] * repetitions
        while completed_experiments < repetitions:
            #env = PlusMazeOneHotCues2ActiveDoors(relevant_cue=CueType.ODOR)
            env.init()
            (brain, learner, model) = agent_spec
            agent = MotivatedAgent(brain(learner(model(env.stimuli_encoding_size(), 2, env.num_actions()),
                                                 learning_rate=config.LEARNING_RATE), batch_size=config.BATCH_SIZE),
                                   motivation=RewardType.WATER, motivated_reward_value=config.MOTIVATED_REWARD,
                                   non_motivated_reward_value=config.NON_MOTIVATED_REWARD)
            experiment_stats, experiment_data = PlusMazeExperiment(env, agent, dashboard=False)
            experiment_data['model'] = utils.brain_name(architecture)
            experiment_data['subject'] = completed_experiments

            if experiment_stats.metadata['experiment_status'] == ExperimentStatus.COMPLETED:
                brain_repetition_reports[completed_experiments] = experiment_stats
                completed_experiments += 1
                all_experiment_data = all_experiment_data.append(experiment_data, ignore_index=True)
            else:
                aborted_experiments += 1
        brains_reports.append(brain_repetition_reports)
        print("{} out of {} experiments were aborted".format(aborted_experiments,
                                                             aborted_experiments + completed_experiments))

    plot_days_per_stage(brains_reports, file_path=os.path.join('Results', 'days_per_stage.png'))

    for brain_report in brains_reports:
        plot_behavior_results(brain_report)


if __name__ == '__main__':
    env = PlusMazeOneHotCues2ActiveDoors(relevant_cue=CueType.ODOR, stimuli_encoding=8)
    #env = PlusMazeOneHotCues(relevant_cue=CueType.ODOR, stimuli_encoding=10)
    run_simulation(env)
    x = 1