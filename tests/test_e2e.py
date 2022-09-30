__author__ = 'gkour'

from environment import PlusMazeOneHotCues, CueType
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


if __name__ == '__main__':
    env = PlusMazeOneHotCues(relevant_cue=CueType.ODOR)

    test_brain(TDBrain(IALearner(learning_rate=0.01,
								 model=FTable(env.stimuli_encoding_size(), 2,
                                         env.num_actions()))))
    test_brain(TDBrain(IALearner(learning_rate=0.01,
								 model=QTable(env.stimuli_encoding_size(), 2, env.num_actions()))))
    test_brain(ConsolidationBrain(DQN(learning_rate=0.01,
                                          model=FCNet(env.stimuli_encoding_size(), 2,
													  env.num_actions()))))