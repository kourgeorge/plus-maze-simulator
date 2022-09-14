__author__ = 'gkour'

import numpy as np
from Stats import Stats
from rewardtype import RewardType
from ReplayMemory import ReplayMemory
from brains.consolidationbrain import ConsolidationBrain


class FittingStats(Stats):

    def dataframe_report(self, trial, report):
        result = super().dataframe_report(trial, report)
        result.update([('Likelihood', report.likelihood),
                       ('CorrectNetwork', report.correct_network)])
        return result

    def _create_report_from_memory(self, experience: ReplayMemory, brain:ConsolidationBrain, last:int):
        report_dict = super()._create_report_from_memory(experience, brain, last)
        last_exp = experience.last(last)
        infos = [data[6] for data in last_exp]
        report_dict.likelihood = np.mean([info.likelihood for info in infos])
        correct_network = [1 if info.network_outcome != RewardType.NONE else 0 for info in infos]
        report_dict.correct_network = np.mean(correct_network)

        return report_dict
