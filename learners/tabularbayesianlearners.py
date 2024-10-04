
# Add the following code to tabularlearners.py
import numpy as np

from learners.abstractlearner import AbstractLearner
from models.tabularbayesianmodels import BayesianFTable, BayesianACFTable


class BayesianFeatureLearner(AbstractLearner):
    """
    Learner that performs Bayesian updates for feature values.
    Works with BayesianFTable.
    """

    def __init__(self, model: BayesianFTable, observation_variance=10, *args, **kwargs):
        super().__init__(model=model, optimizer={'bayesian'}, *args, **kwargs)
        self.observation_variance = observation_variance  # Variance of the reward signal

    def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation):
        actions = np.argmax(action_batch, axis=1)
        selected_cues = self.model.get_selected_action_stimuli(state_batch, actions)

        for idx, (cue_values, r) in enumerate(zip(selected_cues, reward_batch)):
            for col, dim in enumerate(self.model.env_dimensions):
                feature = cue_values[col]
                if feature != self.model.encoding_size:
                    self.update_stimulus_value(dim, feature, motivation, r)

        return {"delta": None}  # Delta is not directly computed here

    def update_stimulus_value(self, dimension, feature, motivation, r):
        # Retrieve prior parameters
        mu_prior = self.model.Q[motivation.value][dimension]['mu'][feature]
        sigma2_prior = self.model.Q[motivation.value][dimension]['sigma2'][feature]

        # Observation noise variance
        sigma_r2 = self.observation_variance

        # Compute Kalman gain
        K = sigma2_prior / (sigma2_prior + sigma_r2)

        # Update mean
        mu_post = mu_prior + K * (r - mu_prior)

        # Update variance
        sigma2_post = (1 - K) * sigma2_prior

        # Store updated parameters
        self.model.set_stimulus_value(dimension, feature, motivation, mu_post, sigma2_post)



class BayesianAttentionLearner(BayesianFeatureLearner):
    """
    Learner that performs Bayesian updates for both feature values and attention weights.
    Works with BayesianACFTable.
    """

    def __init__(self, model: BayesianACFTable, attention_beta=0.1, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        self.attention_beta = attention_beta  # Learning rate for attention updates

    def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation):
        actions = np.argmax(action_batch, axis=1)
        selected_cues = self.model.get_selected_action_stimuli(state_batch, actions)
        batch_size = len(state_batch)

        delta_alpha_total = np.zeros(len(self.model.env_dimensions))

        for idx in range(batch_size):
            cue_values = selected_cues[idx]
            r = reward_batch[idx]

            # Compute expected feature values (means)
            V = np.array([
                self.model.get_stimulus_value(dim, cue_values[i], motivation)
                for i, dim in enumerate(self.model.env_dimensions)
            ])

            # Compute attention weights
            phi = self.model.phi()

            # Compute action value Q(s, a)
            Q_value = np.dot(phi, V)

            # Compute prediction error
            delta = r - Q_value

            # Update feature values using Bayesian update
            for i, dim in enumerate(self.model.env_dimensions):
                feature = cue_values[i]
                if feature != self.model.encoding_size:
                    self.update_stimulus_value(dim, feature, motivation, r)

            # Compute attention weight updates
            delta_alpha = self.compute_attention_delta(V, Q_value, delta)
            delta_alpha_total += delta_alpha

        # Update attention weights
        self.model.update_attention_weights(delta_alpha_total / batch_size)

        return {"delta": np.mean(delta)}

    def compute_attention_delta(self, V, Q, delta):
        # Compute gradient of Q(s,a) with respect to phi^d
        grad_Q_phi = V

        # Update Dirichlet parameters proportional to delta * grad_Q_phi
        delta_alpha = self.attention_beta * delta * grad_Q_phi

        # Ensure parameters remain valid
        delta_alpha = np.maximum(delta_alpha, -self.model.attn_alpha + self.model.minimum_alpha)

        return delta_alpha
