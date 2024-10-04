import numpy as np

import config
import utils
from models.tabularmodels import ACFTable
from rewardtype import RewardType


class BayesianFTable(ACFTable):
    """
    Extends FTable to implement Bayesian updates for feature values.
    Each feature value is represented by a Gaussian distribution with mean and variance.
    """

    def __init__(self, encoding_size, num_actions, initial_value=config.INITIAL_FEATURE_VALUE,
                 initial_variance=1.0, observation_variance=1.0, env_dimensions=['odor', 'color', 'spatial'], *args, **kwargs):
        self.initial_variance = initial_variance
        self.observation_variance = observation_variance  # Variance of the reward signal
        super().__init__(encoding_size, num_actions, initial_value, env_dimensions, *args, **kwargs)
        self.initialize_state_values()

    def initialize_state_values(self):
        for motivation in RewardType:
            self.Q[motivation.value] = dict()
            for stimuli in self.env_dimensions:
                # Initialize mean and variance for each feature
                self.Q[motivation.value][stimuli] = {
                    'mu': self.initial_value * np.ones([self.encoding_size + 1]),
                    'sigma2': self.initial_variance * np.ones([self.encoding_size + 1])
                }

    def get_stimulus_value(self, dimension, feature, motivation):
        # Return the mean value of the feature
        return self.Q[motivation.value][dimension]['mu'][feature]

    def get_stimulus_variance(self, dimension, feature, motivation):
        # Return the variance of the feature
        return self.Q[motivation.value][dimension]['sigma2'][feature]

    def set_stimulus_value(self, dimension, feature, motivation, mu_new, sigma2_new):
        # Update the mean and variance of the feature
        self.Q[motivation.value][dimension]['mu'][feature] = mu_new
        self.Q[motivation.value][dimension]['sigma2'][feature] = sigma2_new

    def get_observations_values(self, observations, motivation):
        """
        Calculate action values based on the expected means of the features.
        """
        cues = utils.stimuli_1hot_to_cues(observations, self.encoding_size)
        batch_size = observations.shape[0]

        vector = np.arange(self._num_actions)
        actions = vector[np.newaxis, :]

        # Add the door (action) cue
        cues = np.concatenate([cues, actions[np.newaxis, :]], axis=1)

        # Stack all stimulus means into a single data array
        data = np.stack([
            self.get_stimulus_value(dim, cues[:, i], motivation)
            for i, dim in enumerate(self.env_dimensions)
        ])

        # Sum over dimensions to get action values
        action_values = np.sum(data, axis=0)

        return action_values

    def new_stimuli_context(self, motivation):
        self.initialize_state_values()


class BayesianACFTable(BayesianFTable):
    """
    Extends BayesianFTable to include Bayesian updates for attention weights.
    Attention weights are modeled using a Dirichlet distribution.
    """

    def __init__(self, initial_alpha, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_alpha = initial_alpha #kwargs.get('initial_alpha', 1.0)
        self.minimum_alpha = kwargs.get('minimum_alpha', 1e-3)
        self.attn_alpha = np.ones(len(self.env_dimensions)) * self.initial_alpha
        self.attn_importance = self.attn_alpha  # Expected value under Dirichlet

    def phi(self):
        # Expected value of attention weights under Dirichlet distribution
        alpha_sum = np.sum(self.attn_alpha)
        return self.attn_alpha / alpha_sum

    def update_attention_weights(self, delta_alpha):
        # Update Dirichlet parameters
        self.attn_alpha += delta_alpha

        # Ensure parameters remain positive
        self.attn_alpha = np.maximum(self.attn_alpha, self.minimum_alpha)

        # # Update attention importance (old alternative)
        # self.attn_importance = self.attn_alpha

        # Update attention importance
        self.attn_importance = self.attn_alpha / np.sum(self.attn_alpha)
