__author__ = 'gkour'

from environment import PlusMazeOneHotCues, CueType, PlusMazeOneHotCues2ActiveDoors, StagesTransition
from fitting.InvestigateAnimal import writecsvfiletotemp
from fitting.MazeResultsBehavioural import model_parameters_development, compare_model_subject_learning_curve_average, \
    investigate_regret_delta_relationship
from rewardtype import RewardType
from models.networkmodels import *
from brains.fixeddoorattentionbrain import FixedDoorAttentionBrain
from learners.networklearners import *
from learners.tabularlearners import *
from brains.consolidationbrain import ConsolidationBrain
from motivatedagent import MotivatedAgent
from PlusMazeExperiment import PlusMazeExperiment
from models.tabularmodels import FTable, QTable
from brains.tdbrain import TDBrain


def test_BasicDQNBrain():
    env = PlusMazeOneHotCues(relevant_cue=CueType.ODOR)
    network = FC2LayersNet(env.stimuli_encoding_size(), 2, env.num_actions())
    learner = DQN(network)
    brain = ConsolidationBrain(learner=learner)

    agent = MotivatedAgent(brain,motivation=RewardType.WATER,
                                       motivated_reward_value=1, non_motivated_reward_value=0.3)
    experiment_stats = PlusMazeExperiment(agent, dashboard=False)

    assert experiment_stats is not None


def test_FixedDoorAttentionBrain():
    env = PlusMazeOneHotCues(relevant_cue=CueType.ODOR)
    network = EfficientNetwork(env.stimuli_encoding_size(), 2, env.num_actions())
    learner = DQN(network)
    brain = FixedDoorAttentionBrain(learner=learner)

    agent = MotivatedAgent(brain,motivation=RewardType.WATER,
                                       motivated_reward_value=1, non_motivated_reward_value=0.3)
    experiment_stats = PlusMazeExperiment(env, agent, dashboard=False)

    assert experiment_stats is not None


def test_brain(brain):
    env = PlusMazeOneHotCues(relevant_cue=CueType.ODOR)
    agent = MotivatedAgent(brain, motivation=RewardType.WATER,
                                       motivated_reward_value=1, non_motivated_reward_value=0.3)
    experiment_stats = PlusMazeExperiment(env, agent, dashboard=False)

    assert experiment_stats is not None


def test_PlusMazeOneHotCues2ActiveDoors():
    stages = [{'name': 'Odor1', 'transition_logic': StagesTransition.set_odor_stage},
              {'name': 'Odor2', 'transition_logic': StagesTransition.set_odor_stage},
              # {'name': 'Odor4', 'transition_logic': StagesTransition.set_odor_stage},
              {'name': 'LED', 'transition_logic': StagesTransition.set_color_stage},
              #{'name': 'LED', 'transition_logic': StagesTransition.set_color_stage},
              ]
    env = PlusMazeOneHotCues2ActiveDoors(stages=stages, relevant_cue=CueType.ODOR, stimuli_encoding=14)

    model = ACFTable(env.stimuli_encoding_size(), 2, env.num_actions())
    learner = MALearner(model, alpha_phi=0.11, learning_rate=0.057)
    brain = TDBrain(learner=learner, beta=5.2)

    agent = MotivatedAgent(brain, motivation=RewardType.NONE,
                                       motivated_reward_value=1, non_motivated_reward_value=0, exploration_param=0.1)
    stats, experiment_data = PlusMazeExperiment(env, agent, dashboard=False)
    experiment_data['model'] = 'AARL'
    experiment_data['subject'] = -1
    experiment_data['parameters'] = -1

    filename = writecsvfiletotemp(experiment_data)
    model_parameters_development(filename,  reward_dependant_trials=None)
    compare_model_subject_learning_curve_average(filename)
    investigate_regret_delta_relationship(filename)

    assert stats is not None

if __name__ == '__main__':
    # env = PlusMazeOneHotCues(relevant_cue=CueType.ODOR)
    #
    # test_brain(TDBrain(IALearner(learning_rate=0.01,
	# 							 model=FTable(env.stimuli_encoding_size(), 2,
    #                                      env.num_actions()))))
    # test_brain(TDBrain(IALearner(learning_rate=0.01,
	# 							 model=QTable(env.stimuli_encoding_size(), 2, env.num_actions()))))
    # test_brain(ConsolidationBrain(DQN(learning_rate=0.01,
    #                                       model=FCNet(env.stimuli_encoding_size(), 2,
	# 												  env.num_actions()))))

    test_PlusMazeOneHotCues2ActiveDoors()