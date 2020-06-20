import numpy as np
import matplotlib.pyplot as plt


class Dashboard:
    def __init__(self, brain):
        self.fig = plt.figure(figsize=(9, 5), dpi=120, facecolor='w')
        self.textstr = "Stage:{}, Trial {}: Odors:{}, Lights:{}. CorrectCue: {}"
        axis_l1 = self.fig.add_subplot(211)
        axis_l2 = self.fig.add_subplot(212)
        self.im1_obj = axis_l1.imshow(np.transpose(brain.policy.l1.weight.data.numpy()), cmap='RdBu', vmin=-2, vmax=2)
        self.im2_obj = axis_l2.imshow(brain.policy.l2.weight.data.numpy(), cmap='RdBu', vmin=-2, vmax=2)
        props = dict(boxstyle='round', facecolor='wheat')
        self.figtxt = plt.figtext(0.1, 0.95, 'Start', fontsize=8, verticalalignment='top', bbox=props)
        self.fig.colorbar(self.im1_obj, ax=axis_l1)
        self.fig.colorbar(self.im2_obj, ax=axis_l2)

    def update(self, trial, report, env, brain):
        textstr = self.textstr.format(
            env.stage, trial, env._odor_options, env._light_options, env._correct_cue_value,
            round(np.mean(report.reward), 2),
            round(np.mean(report.correct), 2)
        )
        self.figtxt.set_text(textstr)
        self.im1_obj.set_data(np.transpose(brain.policy.l1.weight.data.numpy()))
        self.im2_obj.set_data(brain.policy.l2.weight.data.numpy())

    def get_fig(self):
        return self.fig
