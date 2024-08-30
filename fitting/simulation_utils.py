import numpy as np
import pandas as pd

from fitting.InvestigateAnimal import string2list


def extract_model_average_fitting_parameters(fitting_data_df_file, model):
    fitting_data_df = pd.read_csv(fitting_data_df_file)
    fitting_data_df = fitting_data_df[fitting_data_df['model'] == model]
    fitting_data_subject_params_df = fitting_data_df.groupby(['model', 'subject'])['parameters'].first().reset_index()
    fitting_data_subject_params_df['parameters'] = fitting_data_subject_params_df['parameters'].apply(string2list)

    average_parameters = fitting_data_subject_params_df['parameters'].apply(pd.Series).mean().tolist()
    std_parameters = fitting_data_subject_params_df['parameters'].apply(pd.Series).std().tolist()

    return average_parameters, std_parameters


def log_uniform(low=1e-9, high=1e9, size=None, base=10):
    low_log = np.log10(low)
    high_log = np.log10(high)
    return np.power(base, np.random.uniform(low_log, high_log, size))


def sample_brain_parameters(parameters_mean, parameters_std, ranges):
    """Given the parameter mean and std, samples a parameter value within the given range"""
    sampled_simulation_params = [np.random.normal(loc=mean, scale=std) for mean, std in
                                 zip(parameters_mean, parameters_std)]

    estimated_parameters = tuple(np.clip(sampled_simulation_params[i], *ranges[i]) for i, parameter_value in
                                 enumerate(sampled_simulation_params))

    return estimated_parameters
