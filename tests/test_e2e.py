__author__ = 'gkour'

from environment import PlusMazeOneHotCues, CueType
from rewardtype import RewardType
from standardbrainnetwork import FullyConnectedNetwork, EfficientNetwork, SeparateMotivationAreasNetwork, \
    FullyConnectedNetwork2Layers, TabularQ, TabularAL
from fixeddoorattentionbrain import FixedDoorAttentionBrain
from learner import DQN, TDAL, TD
from consolidationbrain import ConsolidationBrain
from motivatedagent import MotivatedAgent
from PlusMazeExperiment import PlusMazeExperiment
from tdbrain import TDBrain


def test_BasicDQNBrain():
    env = PlusMazeOneHotCues(relevant_cue=CueType.ODOR)
    network = FullyConnectedNetwork2Layers(env.stimuli_encoding_size(), 2, env.num_actions())
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
    experiment_stats = PlusMazeExperiment(agent, dashboard=False)

    assert experiment_stats is not None


def test_tabularBrain():
    env = PlusMazeOneHotCues(relevant_cue=CueType.ODOR)
    learner = TD(learning_rate=0.01, model=TabularQ(env.stimuli_encoding_size(), 2, env.num_actions()))
    brain = TDBrain(learner, env.num_actions())

    agent = MotivatedAgent(brain,motivation=RewardType.WATER,
                                       motivated_reward_value=1, non_motivated_reward_value=0.3)
    experiment_stats = PlusMazeExperiment(agent, dashboard=False)

    assert experiment_stats is not None


def test_tabularBrainNiv():
    env = PlusMazeOneHotCues(relevant_cue=CueType.ODOR)
    learner = TDAL(learning_rate=0.01, model=TabularAL(env.stimuli_encoding_size(), 2, env.num_actions()))
    brain = TDBrain(learner, env.num_actions(), batch_size=1)

    agent = MotivatedAgent(brain,motivation=RewardType.WATER,
                                       motivated_reward_value=1, non_motivated_reward_value=0.3)
    experiment_stats = PlusMazeExperiment(agent, dashboard=False)

    assert experiment_stats is not None


if __name__ == '__main__':
    test_FixedDoorAttentionBrain()
    test_tabularBrain()
    test_tabularBrainNiv()