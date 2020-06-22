import numpy as np
import matplotlib.pyplot as plt

#plt.ion()


class Dashboard:
    def __init__(self, brain):
        self.stage = 1
        self.fig = plt.figure(figsize=(9, 5), dpi=120, facecolor='w')
        self.textstr = "Stage:{}, Trial {}: Odors:{}, Lights:{}. CorrectCue: {}. Accuracy:{}, Reward: {}."
        axis_l1 = self.fig.add_subplot(222)
        axis_l2 = self.fig.add_subplot(221)
        self.im1_obj = axis_l1.imshow(np.transpose(brain.policy.affine.weight.data.numpy()), cmap='RdBu', vmin=-2, vmax=2)
        self.im2_obj = axis_l2.imshow(brain.policy.controller.weight.data.numpy(), cmap='RdBu', vmin=-2, vmax=2)
        props = dict(boxstyle='round', facecolor='wheat')
        self.figtxt = plt.figtext(0.1, 0.95, 'Start', fontsize=8, verticalalignment='top', bbox=props)
        self.fig.colorbar(self.im1_obj, ax=axis_l1)
        self.fig.colorbar(self.im2_obj, ax=axis_l2)

        self._axes_graph = self.fig.add_subplot(212)
        self._axes_graph.set_ylabel('Percent')
        self._line_correct, = self._axes_graph.plot([], [], 'g+-', label='Correct', alpha=0.2)
        self._line_reward, = self._axes_graph.plot([], [], 'y-', label='Reward', alpha=0.2)
        self._line_water_preference, = self._axes_graph.plot([], [], '^-', label='Water PI', alpha=0.2)
        self._line_water_correct, = self._axes_graph.plot([], [], 'bo-', label='Water Correct')
        self._line_food_correct, = self._axes_graph.plot([], [], 'ro-', label='Food Correct')



        self._axes_graph.set_ylim(0, 1)

        self._axes_graph.legend(
            [self._line_correct, self._line_reward, self._line_water_correct, self._line_food_correct, self._line_water_preference],
            [self._line_correct.get_label(), self._line_reward.get_label(), self._line_water_correct.get_label(),
             self._line_food_correct.get_label(),  self._line_water_preference.get_label()], loc=0)

 #       plt.show()

    def update(self, stats_df, env, brain):
        textstr = self.textstr.format(
            env.stage, stats_df['Trial'].to_numpy()[-1], env._odor_options, env._light_options, env._correct_cue_value,
            stats_df['Correct'].to_numpy()[-1],
            stats_df['Reward'].to_numpy()[-1]
        )

        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()

        self.figtxt.set_text(textstr)
        self.im1_obj.set_data(np.transpose(brain.policy.affine.weight.data.numpy()))
        self.im2_obj.set_data(brain.policy.controller.weight.data.numpy())

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


        if self.stage < env.stage:
            self.stage = env.stage
            self._axes_graph.axvline(x=stats_df['Trial'].to_numpy()[-1]-50, alpha=0.5, dashes=(5, 2, 1, 2), lw=2)

        self._axes_graph.relim()
        self._axes_graph.autoscale_view()

    def get_fig(self):
        return self.fig
