import pandas as pd

import config
import matplotlib.pyplot as plt
import seaborn as sns

from fitting import fitting_utils


def analyze_parameters_aloowing_task_completion(df):
    # Step 1: Calculate the last stage and last day for each subject
    last_stage = df.groupby('subject')['stage'].max().reset_index()
    last_stage.columns = ['subject', 'last_stage']

    # Find the last day in each stage
    last_day_in_stage = df.groupby(['subject', 'stage'])['day in stage'].max().reset_index()
    last_day_in_stage.columns = ['subject', 'stage', 'last_day']

    # Merge to get the last day of the last stage for each subject
    last_stage_day = last_day_in_stage.merge(last_stage, left_on=['subject', 'stage'],
                                             right_on=['subject', 'last_stage'])
    last_stage_day = last_stage_day.drop(columns=['last_stage'])

    # Step 2: Calculate success rate on the last day of the last stage
    # Filter df to include only the last stage data
    last_stage_data = df.merge(last_stage_day[['subject', 'stage', 'last_day']],
                               left_on=['subject', 'stage', 'day in stage'],
                               right_on=['subject', 'stage', 'last_day'])

    # Determine if task completion criterion (75% reward rate) is met
    # Add condition to check if the last stage is 3
    task_completion = last_stage_data.groupby('subject').agg(
        last_stage=('stage', 'max'),
        mean_reward=('reward', 'mean')
    ).reset_index()

    task_completion['task_completion'] = (task_completion['last_stage'] == 3) & (task_completion['mean_reward'] >= 0.75)
    task_completion = task_completion[['subject', 'task_completion']]

    # Step 3: Add model and parameters to the results
    model_params = df[['subject', 'model', 'parameters']].drop_duplicates().reset_index(drop=True)
    result = task_completion.merge(model_params, on='subject')

    # Add the success rate to the final result
    success_rate = last_stage_data.groupby('subject')['reward'].mean().reset_index(name='success_rate')
    result = result.merge(success_rate, on='subject')

    # Reorder columns
    result = result[['subject', 'model', 'parameters', 'task_completion', 'success_rate']]

    return result

def plot_task_completion_parameters(results):
    # Separate data based on task completion status

    results['parameters'] = results.parameters.apply(lambda x: fitting_utils.string2list(x))
    markers = ['o', 's', '^', 'D', 'v', '<', '>']  # Define as many markers as you have models
    colors = {'completed': 'blue', 'not completed': 'red'}

    models = results.model.unique()

    plt.figure(figsize=(10, 6))

    for model in models:
        model_data = results[results['model'] == model]

        # Separate completed and not completed
        completed = model_data[model_data['task_completion']]
        not_completed = model_data[~model_data['task_completion']]


        # Plot parameter combinations that allowed task completion
        for params in completed['parameters']:
            plt.scatter(params[0], params[1], color='blue', marker=markers[models.tolist().index(model)],
                        label='Task Completion' if 'Task Completion' not in plt.gca().get_legend_handles_labels()[
                            1] else "")

        # Plot parameter combinations that did not allow task completion
        for params in not_completed['parameters']:
            plt.scatter(params[0], params[1], color='red', marker=markers[models.tolist().index(model)],
                        label='No Task Completion' if 'No Task Completion' not in plt.gca().get_legend_handles_labels()[
                            1] else "")

    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\alpha$')
    plt.title('Parameter Combinations for Task Completion')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    all_simulation_data = pd.read_csv('fitting/Results/Rats-Results/identifiability_results/simulation_25_100TPD_loguniform_original_range_failing_allowed.csv')
    results = analyze_parameters_aloowing_task_completion(all_simulation_data)
    plot_task_completion_parameters(results)