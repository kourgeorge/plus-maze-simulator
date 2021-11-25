import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


def plot_days_per_stage(reports_pg, reports_dqn):
    stages = list(range(1, 6))
    repetitions = len(reports_pg)
    days_per_stage_pg = []
    days_per_stage_dqn = []
    for experiment_report_df_pg in reports_pg:
        c = Counter(list(experiment_report_df_pg['Stage']))
        days_per_stage_pg.append([c[i] for i in stages])

    for experiment_report_df_dqn in reports_dqn:
        c = Counter(list(experiment_report_df_dqn['Stage']))
        days_per_stage_dqn.append([c[i] for i in stages])

    days_per_stage_pg = np.stack(days_per_stage_pg)
    days_per_stage_dqn = np.stack(days_per_stage_dqn)

    width = 0.25
    X = np.array(stages)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(stages, np.mean(days_per_stage_pg, axis=0), yerr=np.std(days_per_stage_pg, axis=0), color='b', width=width, label='PG', capsize=2)
    ax.bar(np.array(stages) + width, np.mean(days_per_stage_dqn, axis=0), yerr=np.std(days_per_stage_dqn, axis=0), color='g', width=width, label='DQN', capsize=2)
    plt.title("Days Per stage - PG vs DQN. #reps={}".format(repetitions))
    plt.legend()
    plt.show()


def days_to_consider_in_each_stage(reports):
    stages = list(range(1, 6))
    days_per_stage = []
    for experiment_report_df in reports:
        c = Counter(list(experiment_report_df['Stage']))
        days_per_stage.append([c[i] for i in stages])


