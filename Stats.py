__author__ = 'gkour'

import numpy as np
import pandas as pd
from collections import OrderedDict
import utils
import config
from ReplayMemory import ReplayMemory
from torchbrain import TorchBrain
import copy
from motivatedagent import MotivatedAgent


class Stats:

    def __init__(self, metadata=None):
        self.action_dist = np.zeros(4)
        self.reports = []
        self.epoch_stats_df = pd.DataFrame()
        self.metadata = metadata

    def update_stats(self,agent:MotivatedAgent, trial, last):
        report = Stats.create_report_from_memory(agent.get_memory(), agent.get_brain(), last)
        self.reports += [report]
        stats = self.dataframe_report(trial, report)
        temp_df = pd.DataFrame([stats], columns=stats.keys())
        self.epoch_stats_df = self.epoch_stats_df.append(temp_df, ignore_index=True)

    def dataframe_report(self, trial, report):
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
            ('AffineDim', utils.network_diff(self.reports[-1].brain, self.reports[-2].brain if len(self.reports) > 2 else self.reports[-1].brain)),
            ('ControllerDim', report.controller_dim)])

    @staticmethod
    def create_report_from_memory(experience: ReplayMemory, brain: TorchBrain, last):
        if last == -1:
            pass
        last_exp = experience.last(last)
        actions = [data[1] for data in last_exp]
        rewards = [data[2] for data in last_exp]
        infos = [data[6] for data in last_exp]

        report_dict = utils.Object()
        report_dict.action_1hot = actions
        report_dict.action = np.argmax(actions, axis=1)
        report_dict.reward = rewards
        report_dict.arm_type_water = [1 if action < 2 else 0 for action in report_dict.action]
        report_dict.correct = [1 if info.outcome != config.RewardType.NONE else 0 for info in infos]
        report_dict.stage = [info.stage for info in infos]
        report_dict.brain = copy.deepcopy(brain)
        report_dict.affine_dim, report_dict.controller_dim = utils.electrophysiology_analysis(brain)
        return report_dict
