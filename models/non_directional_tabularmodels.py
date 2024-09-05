import numpy as np

from models.tabularmodels import OptionsTable, ACFTable
import utils


class NonDirectionalOptionsTable(OptionsTable):
    def __init__(self, *args, **kwargs):
        super().__init__(use_location_cue=False, *args, **kwargs)


class NonDirectionalACFTable(ACFTable):
    def __init__(self, *args, **kwargs):
        super().__init__(env_dimensions=['odors', 'colors'], *args, **kwargs)

    def get_observations_values(self, observations, motivation):
        # Convert one-hot encoded observations to cues (odor and color for each door)
        cues = utils.stimuli_1hot_to_cues(observations, self.encoding_size)
        batch_size = observations.shape[0]

        # Extract odors and colors from cues
        odor_cues = cues[:, 0]  # Odor for each door
        color_cues = cues[:, 1]  # Color for each door

        # Calculate stimulus values for odors, colors, and spatial dimensions
        odor_values = self.get_stimulus_value('odors', odor_cues, motivation)
        color_values = self.get_stimulus_value('colors', color_cues, motivation)

        # Stack all stimulus values into a single data array
        data = np.stack([odor_values, color_values])

        # Calculate the attention weights (phi) and apply them to the data
        attention_weights = np.expand_dims(self.phi(), axis=0)
        action_values = np.matmul(attention_weights, np.transpose(data, axes=(1, 0, 2)))

        # Squeeze the action values to remove the singleton dimension
        action_values = np.squeeze(action_values, axis=1)

        return action_values


class NonDirectionalFixedACFTable(NonDirectionalACFTable):
    """Similar to ACFTable, but the attention weights can be initialized and fixed.
    It suggests that attention may not change dynamically during learning."""

    def __init__(self, attn_importance=np.ones([2]) / 2, *args, **kwargs):
        if not utils.is_valid_attention_weights(attn_importance):
            raise Exception("Illegal attention arguments, should be positive and sum to 1!")
        super().__init__(*args, **kwargs)

        self.attn_importance = np.abs(attn_importance) / np.sum(
            np.abs(attn_importance))  # normalize attention parameters

    def phi(self):
        return self.attn_importance