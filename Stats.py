import numpy as np
import pandas as pd
from collections import OrderedDict


class Stats:

    def __init__(self):
        self.action_dist = np.zeros(4)
        self.epoch_stats_df = pd.DataFrame()

    def update(self, trial, report):
        stats = self.collect_stats(trial, report)
        temp_df = pd.DataFrame([stats], columns=stats.keys())
        self.epoch_stats_df = pd.concat([self.epoch_stats_df, temp_df], axis=0).reset_index(drop=True)

        return self.epoch_stats_df

    def collect_stats(self, trial, report):
        action_dist = np.mean(report.action_1hot, axis=0)
        avg_correct = np.mean(report.correct)
        avg_reward = round(np.mean(report.reward), 2)

        water_preference = round(np.sum(report.arm_type_water) / len(report.arm_type_water), 2)
        water_correct_percent = round(
            np.sum(np.logical_and(report.arm_type_water, report.correct)) /
            np.sum(report.arm_type_water), 2)
        food_correct_percent = round(
            np.sum(np.logical_and(np.logical_not(report.arm_type_water), report.correct)) /
            np.sum(np.logical_not(report.arm_type_water)), 2)

        return OrderedDict([
            ('Trial', trial),
            ('Stage', report.stage[-1]),
            ('ActionDist', action_dist),
            ('Correct', avg_correct),
            ('Reward', avg_reward),
            ('WaterPreference', water_preference),
            ('WaterCorrect', water_correct_percent),
            ('FoodCorrect', food_correct_percent),
            ('AffineDim', report.affine_dim),
            ('ControllerDim', report.controller_dim)])

