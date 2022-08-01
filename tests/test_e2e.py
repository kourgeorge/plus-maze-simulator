__author__ = 'gkour'

from environment import PlusMazeOneHotCues
import config
from standardbrainnetwork import FullyConnectedNetwork, EfficientNetwork, SeparateMotivationAreasNetwork, \
    FullyConnectedNetwork2Layers
from fixeddoorattentionbrain import FixedDoorAttentionBrain
from braindqn import DQN
from torchbrain import ConsolidationBrain
from motivatedagent import MotivatedAgent
from PlusMazeExperiment import PlusMazeExperiment
import torch.optim as optim


def test_BasicDQNBrain():
    env = PlusMazeOneHotCues(relevant_cue=config.CueType.ODOR)
    network = FullyConnectedNetwork2Layers(env.stimuli_encoding_size(), 2, env.num_actions())
    learner = DQN(network)
    brain = ConsolidationBrain(learner=learner)

    agent = MotivatedAgent(brain,motivation=config.RewardType.WATER,
                                       motivated_reward_value=1, non_motivated_reward_value=0.3)
    experiment_stats = PlusMazeExperiment(agent, dashboard=False)

    assert experiment_stats is not None


def test_FixedDoorAttentionBrain():
    env = PlusMazeOneHotCues(relevant_cue=config.CueType.ODOR)
    network = EfficientNetwork(env.stimuli_encoding_size(), 2, env.num_actions())
    learner = DQN(network)
    brain =FixedDoorAttentionBrain(learner=learner)

    agent = MotivatedAgent(brain,motivation=config.RewardType.WATER,
                                       motivated_reward_value=1, non_motivated_reward_value=0.3)
    experiment_stats = PlusMazeExperiment(agent, dashboard=False)

    assert experiment_stats is not None