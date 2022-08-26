__author__ = 'gkour'

import numpy as np
import pandas as pd
from collections import OrderedDict
import utils
from rewardtype import RewardType
from ReplayMemory import ReplayMemory
from consolidationbrain import ConsolidationBrain
import copy
from motivatedagent import MotivatedAgent


class Stats:
    """This class accumulate report objects that contains information about each day.
    In addition, it maintains a regularly updated  dataframe that allows consumers read the information in an asier fashion."""
    def __init__(self, metadata=None):
        self.action_dist = np.zeros(4)
        self.reports = []
        self.epoch_stats_df = pd.DataFrame()
        self.metadata = metadata

    def update_stats_from_agent(self, agent:MotivatedAgent, trial, last):
        report = self._create_report_from_memory(agent.get_memory(), agent.get_brain(), last)
        self.reports += [report]
        stats = self.dataframe_report(trial, report)
        temp_df = pd.DataFrame([stats], columns=stats.keys())
        self.epoch_stats_df = self.epoch_stats_df.append(temp_df, ignore_index=True)

    def dataframe_report(self, trial, report):

        return OrderedDict([
                               ('Trial', trial),
                               ('Stage', report.stage),
                               ('ActionDist', report.action_1hot),
                               ('Correct', report.correct),
                               ('Reward', report.reward),
                               ('WaterPreference', report.water_preference),
                               ('WaterCorrect', report.water_correct_percent),
                               ('FoodCorrect', report.food_correct_percent)] +
                           [(k, v) for k, v in self.reports[-1].brain.get_network().network_diff(
                               self.get_last_day_in_previous_stage().brain.get_network()).items()] +
                           [(k, v) for k, v in self.reports[-1].brain.get_network().get_network_metrics().items()])


    def get_last_day_in_previous_stage(self):
        current_stage = self.reports[-1].stage
        if current_stage==0:
            return self.reports[0]

        return [report for report in self.reports if report.stage==current_stage-1][-1]


    def _create_report_from_memory(self, experience: ReplayMemory, brain: ConsolidationBrain, last):
        if last == -1:
            pass
        last_exp = experience.last(last)
        actions = [data[1] for data in last_exp]
        rewards = [data[2] for data in last_exp]
        infos = [data[6] for data in last_exp]

        action = np.argmax(actions, axis=1)
        correct = [1 if info.outcome != RewardType.NONE else 0 for info in infos]
        arm_type_water = [1 if action < 2 else 0 for action in action]


        report_dict = utils.Object()
        report_dict.action_1hot = np.mean(actions, axis=0)
        report_dict.reward = round(np.mean(rewards), 2)
        report_dict.water_preference = round(np.sum(arm_type_water) / len(arm_type_water), 2)
        report_dict.correct = np.mean(correct)
        report_dict.water_correct_percent = round(np.sum(np.logical_and(arm_type_water, correct))/np.sum(arm_type_water), 2)
        report_dict.food_correct_percent = round(
            np.sum(np.logical_and(np.logical_not(arm_type_water), correct)) / np.sum(np.logical_not(arm_type_water)), 2)
        report_dict.stage = [info.stage for info in infos][-1]
        report_dict.brain = copy.deepcopy(brain)
        return report_dict
