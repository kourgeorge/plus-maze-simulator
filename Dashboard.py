__author__ = 'gkour'

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

import pandas as pd

import utils
from environment import PlusMaze
from consolidationbrain import ConsolidationBrain

plt.ion()


class Dashboard:
    def __init__(self, brain:ConsolidationBrain):
        self.stage = 0
        self.stage_initial_brain = brain
        self.fig = plt.figure(figsize=(9, 7), dpi=120, facecolor='w')
        self.text_curr_stage = "Stage:{}, Trial {}: Odors:{}, Lights:{}. CorrectCue: {}. Accuracy:{}, Reward: {}."
        axis_stimuli = self.fig.add_subplot(322)
        axis_door_attn = self.fig.add_subplot(341)
        axis_dim_attn = self.fig.add_subplot(342)
        axis_stimuli.title.set_text('Stimuli Processing')
        axis_door_attn.title.set_text('Door Attention')
        axis_dim_attn.title.set_text('Dimension Attention')
        axis_dim_attn.set_axis_off()
        self.im1_obj = axis_stimuli.imshow(np.transpose(brain.get_model().get_stimuli_layer().data.numpy()), cmap='RdBu', vmin=-2, vmax=2)
        self.im2_obj = axis_door_attn.imshow(brain.get_model().get_door_attention().data.numpy().T, cmap='RdBu', vmin=-2, vmax=2)
        self.im3_obj = axis_dim_attn.imshow(brain.get_model().get_dimension_attention().data.numpy(), cmap='RdBu', vmin=-2, vmax=2)

        props = dict(boxstyle='round', facecolor='wheat')
        self.figtxtbrain = plt.figtext(0.1, 0.98, "Brain - {}:{}({})".format(str(brain), str(brain.get_model()), str(brain.num_trainable_parameters())), fontsize=8, verticalalignment='top', bbox=props)
        self.figtxt = plt.figtext(0.1, 0.95, 'Start', fontsize=8, verticalalignment='top', bbox=props)
        self.fig.colorbar(self.im1_obj, ax=axis_stimuli)

        self._axes_graph = self.fig.add_subplot(312)
        self._axes_graph.set_ylabel('Behavioral [%]')
        self._axes_graph.set_ylim(0, 1)
        self._line_correct, = self._axes_graph.plot([], [], 'g+-', label='Correct', alpha=0.3)
        self._line_reward, = self._axes_graph.plot([], [], 'y-', label='Reward', alpha=0.2)
        self._line_water_preference, = self._axes_graph.plot([], [], '^-', label='Water PI', markersize=3, alpha=0.4)
        self._line_water_correct, = self._axes_graph.plot([], [], 'bo-', label='Water Correct', markersize=3)
        self._line_food_correct, = self._axes_graph.plot([], [], 'ro-', label='Food Correct', markersize=3)

        self._axes_neural_graph = self.fig.add_subplot(313)
        self._axes_neural_graph.set_ylabel('Neural [%]')
        #self._axes_neural_graph.set_ylim(0, 10)
        self._line_brain_signals = []
        all_brain_signals = list(brain.get_model().get_network_metrics().keys()) + list(brain.get_model().network_diff(brain.get_model()).keys())
        for signal_name in all_brain_signals:
            self._line_brain_signals += self._axes_neural_graph.plot([], [], 'o-', color=utils.colorify(signal_name), label=signal_name, markersize=3,
                                                                  alpha=0.4)

        self._line_brain_compare = []
        for signal_name in brain.get_model().network_diff(brain.get_model()):
            self._line_brain_compare += self._axes_neural_graph.plot([], [], '^-', color=utils.colorify(signal_name), label=signal_name, markersize=3,
                                                                  alpha=0.4)

        self._axes_graph.legend(
            [self._line_correct, self._line_reward, self._line_water_correct, self._line_food_correct,
             self._line_water_preference, ],
            [self._line_correct.get_label(), self._line_reward.get_label(), self._line_water_correct.get_label(),
             self._line_food_correct.get_label(), self._line_water_preference.get_label()], loc=0, prop={'size': 5})

        self._axes_neural_graph.legend(
             self._line_brain_signals + self._line_brain_compare,
             [signal_line.get_label() for signal_line in self._line_brain_signals] +
            [signal_line.get_label() for signal_line in self._line_brain_compare], loc=0, prop={'size': 5})

    def update(self, stats_df, env: PlusMaze, brain:ConsolidationBrain):
        textstr = self.text_curr_stage.format(
            env._stage, stats_df['Trial'].to_numpy()[-1], [np.argmax(encoding) for encoding in env.get_odor_cues()],
            [np.argmax(encoding) for encoding in env.get_light_cues()], np.argmax(env.get_correct_cue_value()),
            stats_df['Correct'].to_numpy()[-1],
            stats_df['Reward'].to_numpy()[-1]
        )

        self.figtxt.set_text(textstr)
        self.im1_obj.set_data(np.transpose(brain.network.get_stimuli_layer().data.numpy()))
        self.im2_obj.set_data(brain.network.get_door_attention().data.numpy().T)
        self.im3_obj.set_data(brain.network.get_dimension_attention().data.numpy())

        self._line_correct.set_xdata(stats_df['Trial'])
        self._line_correct.set_ydata(stats_df['Correct'])

        self._line_reward.set_xdata(stats_df['Trial'])
        self._line_reward.set_ydata(stats_df['Reward'])

        self._line_water_correct.set_xdata(stats_df['Trial'])
        self._line_water_correct.set_ydata(stats_df['WaterCorrect'])

        self._line_food_correct.set_xdata(stats_df['Trial'])
        self._line_food_correct.set_ydata(stats_df['FoodCorrect'])

        self._line_water_preference.set_xdata(stats_df['Trial'])
        self._line_water_preference.set_ydata(stats_df['WaterPreference'])


        for line in self._line_brain_signals:
            line.set_xdata(stats_df['Trial'])
            line.set_ydata(stats_df[line.get_label()])

        if self.stage < env._stage:
            self.stage = env._stage
            self.stage_initial_brain = brain
            self._axes_graph.axvline(x=stats_df['Trial'].to_numpy()[-1] - 50, alpha=0.5, dashes=(5, 2, 1, 2), lw=2)
            self._axes_neural_graph.axvline(x=stats_df['Trial'].to_numpy()[-1] - 50, alpha=0.5, dashes=(5, 2, 1, 2), lw=2)


        self._axes_graph.relim()
        self._axes_graph.autoscale_view()

        self._axes_neural_graph.relim()
        self._axes_neural_graph.autoscale_view()

    def get_fig(self):
        return self.fig

    def close(self):
        self.fig.close()

    def save_fig(self, path, stage):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.fig.savefig(os.path.join(path, "Stage: {}".format(stage)))
