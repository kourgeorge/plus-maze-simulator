__author__ = 'gkour'

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from environment import PlusMaze
from abstractbrain import AbstractBrain
plt.ion()


class Dashboard:
    def __init__(self, brain:AbstractBrain):
        self.stage = 0
        self.fig = plt.figure(figsize=(9, 7), dpi=120, facecolor='w')
        self.text_curr_stage = "Stage:{}, Trial {}: Odors:{}, Lights:{}. CorrectCue: {}. Accuracy:{}, Reward: {}."
        axis_affine = self.fig.add_subplot(322)
        axis_actor = self.fig.add_subplot(421)
        axis_affine.title.set_text('Attention')
        axis_actor.title.set_text('Controller')
        self.im1_obj = axis_affine.imshow(np.transpose(brain.get_network().affine.weight.data.numpy()), cmap='RdBu', vmin=-2, vmax=2)
        self.im2_obj = axis_actor.imshow(brain.get_network().controller.weight.data.numpy(), cmap='RdBu', vmin=-2, vmax=2)

        props = dict(boxstyle='round', facecolor='wheat')
        self.figtxtbrain = plt.figtext(0.1, 0.99, "Brain:{}".format(str(brain)), fontsize=8, verticalalignment='top', bbox=props)
        self.figtxt = plt.figtext(0.1, 0.97, 'Start', fontsize=8, verticalalignment='top', bbox=props)
        self.fig.colorbar(self.im1_obj, ax=axis_affine)

        self._axes_graph = self.fig.add_subplot(312)
        self._axes_graph.set_ylabel('Behavioral [%]')
        self._line_correct, = self._axes_graph.plot([], [], 'g+-', label='Correct', alpha=0.3)
        self._line_reward, = self._axes_graph.plot([], [], 'y-', label='Reward', alpha=0.2)
        self._line_water_preference, = self._axes_graph.plot([], [], '^-', label='Water PI', markersize=3, alpha=0.4)
        self._line_water_correct, = self._axes_graph.plot([], [], 'bo-', label='Water Correct', markersize=3)
        self._line_food_correct, = self._axes_graph.plot([], [], 'ro-', label='Food Correct', markersize=3)

        self._axes_neural_graph = self.fig.add_subplot(313)
        self._axes_neural_graph.set_ylabel('Neural [%]')

        self._axes_graph.set_ylim(0, 1)
        self._axes_neural_graph.set_ylim(0, 10)

        self._line_affine_dimensionality, = self._axes_neural_graph.plot([], [], 'm^-', label='Affine Dim', markersize=3,
                                                                  alpha=0.4)

        self._axes_graph.legend(
            [self._line_correct, self._line_reward, self._line_water_correct, self._line_food_correct,
             self._line_water_preference, ],
            [self._line_correct.get_label(), self._line_reward.get_label(), self._line_water_correct.get_label(),
             self._line_food_correct.get_label(), self._line_water_preference.get_label()], loc=0)

        self._axes_neural_graph.legend(
            [self._line_affine_dimensionality],
            [self._line_affine_dimensionality.get_label()], loc=0)

    def update(self, stats_df, env:PlusMaze, brain):
        textstr = self.text_curr_stage.format(
            env._stage, stats_df['Trial'].to_numpy()[-1], env.get_odor_cues(), env.get_light_cues(), env.get_correct_cue_value(),
            stats_df['Correct'].to_numpy()[-1],
            stats_df['Reward'].to_numpy()[-1]
        )

        self.figtxt.set_text(textstr)
        self.im1_obj.set_data(np.transpose(brain.network.affine.weight.data.numpy()))
        self.im2_obj.set_data(brain.network.controller.weight.data.numpy())

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

        self._line_affine_dimensionality.set_xdata(stats_df['Trial'])
        self._line_affine_dimensionality.set_ydata(stats_df['AffineDim'])

        if self.stage < env._stage:
            self.stage = env._stage
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
