__author__ = 'gkour'

import pandas as pd

from PlusMazeExperiment import PlusMazeExperiment, ExperimentStatus
from behavioral_analysis import plot_days_per_stage, plot_behavior_results
from brains.consolidationbrain import ConsolidationBrain
from brains.tdbrain import TDBrain
from config import TRIALS_IN_DAY
from environment import CueType, PlusMazeOneHotCues, PlusMazeOneHotCues2ActiveDoors, StagesTransition, PlusMaze
from fitting import fitting_utils
from fitting.fitting_utils import extract_names_from_architecture, string2list, resolve_parameters
from fitting.simulation_utils import sample_brain_parameters, extract_model_average_fitting_parameters
from learners.networklearners import *
from learners.tabularlearners import *
from models.networkmodels import *
from models.tabularmodels import *
from motivatedagent import MotivatedAgent
from rewardtype import RewardType
from fitting.fitting_config_attention import friendly_models_name_map, get_parameter_names, map_maze_models, maze_models


def run_single_simulation(env, architecture, model_parameters, initial_motivation=RewardType.WATER):
    (brain, learner, model) = architecture
    resolved_params_dict = resolve_parameters(model_parameters, *architecture)

    model_instance = model(encoding_size=env.stimuli_encoding_size(), num_actions=env.num_actions(), num_channels=2,
                           **resolved_params_dict)
    learner_instance = learner(model_instance, **resolved_params_dict)
    brain_instance = brain(learner_instance, **resolved_params_dict)

    agent = MotivatedAgent(brain_instance, motivation=initial_motivation,
                           motivated_reward_value=config.MOTIVATED_REWARD,
                           non_motivated_reward_value=0)
    experiment_stats, experiment_data = PlusMazeExperiment(env, agent, dashboard=False)
    experiment_data['initial_motivation'] = initial_motivation.value
    experiment_data['model'] = utils.brain_name(architecture)
    experiment_data['parameters'] = [model_parameters] * len(experiment_data)

    return experiment_stats, experiment_data


def run_simulation_sampled_brain(env: PlusMaze, brain_specs, repetitions=10,
                                 initial_motivation: RewardType = RewardType.NONE, require_task_completion=False):
    """Given a PlusMaze environment and a set of agent architectures, runs a simulation of the agent on the environment."""

    all_simulation_data = pd.DataFrame()
    brains_reports = []
    agent_id = 0
    for agent_spec in brain_specs:
        completed_experiments = 0
        aborted_experiments = 0
        brain_repetition_reports = [None] * repetitions
        while completed_experiments < repetitions:
            env.init()
            architecture, param_space = agent_spec

            # estimated_parameters = sample_brain_parameters(parameters_mean, parameters_std, param_space)
            sampled_parameters = [param.rvs()[0] for param in param_space]

            # estimated_parameters = sample_from_estimated_parameters(utils.brain_name(architecture))
            print(sampled_parameters)
            experiment_stats, experiment_data = run_single_simulation(env, architecture, sampled_parameters,
                                                                      initial_motivation)
            experiment_data['initial_motivation'] = initial_motivation.value
            experiment_data['true_model'] = utils.brain_name(architecture)
            experiment_data['subject'] = agent_id
            # experiment_data['true_parameters'] = [[np.round(estimated_parameters,4)]]*len(experiment_data)

            if (not require_task_completion) or (require_task_completion and experiment_stats.metadata[
                'experiment_status'] == ExperimentStatus.COMPLETED):
                brain_repetition_reports[completed_experiments] = experiment_stats
                completed_experiments += 1
                agent_id -= 1
                all_simulation_data = pd.concat([all_simulation_data, experiment_data])
            else:
                aborted_experiments += 1
        brains_reports.append(brain_repetition_reports)
        print("{} out of {} experiments were aborted".format(aborted_experiments,
                                                             aborted_experiments + completed_experiments))

    return all_simulation_data


def WPI_simulations():
    models = [((TDBrain, IALearner, FTable), (0, 2.5, 0.01, 0.1)),  # FRL
              ((TDBrain, IALearner, MFTable), (0, 2.5, 0.01, 0.1)),  # S(V)-FRL
              ((TDBrain, UABIALearner, FTable), (0, 2.5, 0.01, 0.1)),  # B-FRL
              ((TDBrain, ABIALearner, FTable), (0, 2.5, 0.01, 0.1)),  # M(B)-FRL
              ]

    # models=[	((TDBrain, QLearner, QTable), (0, 2.5, 0.01, 0.1)), #SARL
    # 			((TDBrain, UABQLearner, QTable), (0, 2.5, 0.01, 0.1)), #B-SARL
    # 			((TDBrain, ABQLearner, QTable), (0, 2.5, 0.01, 0.1)), # M(B)-SARL
    # 			]
    #
    # models=[	#((TDBrain, QLearner, OptionsTable), (0, 2.5, 0.01, 0.1)), #SARL
    #
    # 			((TDBrain, UABQLearner, OptionsTable), (0, 2.5, 0.01, 0.1)), #B-SARL
    # 			#((TDBrain, ABQLearner, OptionsTable), (0, 2.5, 0.01, 0.1)), # M(B)-SARL
    # 			]
    run_simulation_sampled_brain(PlusMazeOneHotCues(relevant_cue=CueType.ODOR, stimuli_encoding=10), models,
                                 repetitions=20)
    # goal_choice_index_model(data_file_path)

    # calcWPI
    """"To capture the goal-choice element, we quantified for each animal in each training session its bias towards one arm pair according to the goal choice index (GC , Fig. 1e), describing the excess of visits to the arms that contained the deprived reward (regardless of whether they made a correct or erroneous choice): GC=\frac{m-um}{m+um} , where m denotes the number of trials that the animal chose an arms in which correct performance would be rewarded deprived reward (i.e., water for water restriction and food for food restriction condition), and um denotes the number of trials where the animal chose one of the two arms which contained the reward that was not restricted (Fig. 1e).  """


def run_dimensional_shifting_simulations_from_fitting_data(fitting_data_df_file, num_repetitions,
                                                           stages=PlusMazeOneHotCues2ActiveDoors.default_stages,
                                                           require_task_completion=True):
    """ Given fitting data, extract all models and their parameters, then run simulations while
    taking into consideration the mean and std of the estimated parameters."""
    fitting_data_df = pd.read_csv(fitting_data_df_file)

    fitting_data_df = fitting_data_df.groupby(['model', 'subject'])['parameters'].first().reset_index()
    # Apply the function to create new columns for learner and model names

    models = []
    for arch in fitting_data_df['model'].unique():
        df = fitting_data_df[fitting_data_df['model'] == arch]
        df['parameters'] = df['parameters'].apply(string2list)

        # Calculate the average of each parameter across all subjects
        average_parameters = df['parameters'].apply(pd.Series).mean().tolist()
        std_parameters = df['parameters'].apply(pd.Series).std().tolist()

        model_arch, parameters_spec = map_maze_models[arch]
        agent_class, learner_class, model_class = model_arch
        bounds = [parameter.bounds for parameter in parameters_spec]
        models += [
            ((TDBrain, learner_class, model_class), tuple(average_parameters), tuple(std_parameters), parameters_spec)]

    env = PlusMazeOneHotCues2ActiveDoors(stages=stages, relevant_cue=CueType.ODOR, stimuli_encoding=8)
    all_simulation_data = run_simulation_sampled_brain(env, models, repetitions=num_repetitions,
                                                       initial_motivation=RewardType.NONE,
                                                       require_task_completion=require_task_completion)

    return all_simulation_data


def run_dimensional_shifting_simulations(num_repetitions, stages=PlusMazeOneHotCues2ActiveDoors.default_stages,
                                         require_task_completion=True):
    models = []
    for curr_model in maze_models:
        model_arch, parameters_space = curr_model
        agent_class, learner_class, model_class = model_arch
        models += [((TDBrain, learner_class, model_class), parameters_space)]

    env = PlusMazeOneHotCues2ActiveDoors(stages=stages, relevant_cue=CueType.ODOR, stimuli_encoding=8)
    all_simulation_data = run_simulation_sampled_brain(env, models, repetitions=num_repetitions,
                                                       initial_motivation=RewardType.NONE,
                                                       require_task_completion=require_task_completion)

    return all_simulation_data


def run_increasing_IDShift(fitting_data_df_file, repetitions=50):
    average_parameters, std_parameters = extract_model_average_fitting_parameters(fitting_data_df_file,
                                                                                  'MALearner.ACFTable')
    stages = [{'name': 'LED', 'transition_logic': StagesTransition.set_color_stage}]
    all_simulation_data = pd.DataFrame()
    for i in range(6):
        env = PlusMazeOneHotCues2ActiveDoors(stages=stages, relevant_cue=CueType.ODOR, stimuli_encoding=14)

        env_df = run_simulation_sampled_brain(env=env, brain_specs=[((TDBrain, MALearner, ACFTable),
                                                                     average_parameters, std_parameters,
                                                                     ([0.1, 10], [0.001, 0.4], [0.001, 0.4]))],
                                              repetitions=repetitions, initial_motivation=RewardType.NONE)

        env_df["env_setup"] = ",".join([stage['name'] for stage in stages])
        all_simulation_data = pd.concat([all_simulation_data, env_df])
        stages.insert(0, {'name': f'Odor{i + 1}', 'transition_logic': StagesTransition.set_odor_stage})

    return all_simulation_data


def ED_shift_analysis(fitting_file_name, repetitions=50):
    stages = [{'name': 'Odor', 'transition_logic': StagesTransition.set_odor_stage},
              {'name': 'LED1', 'transition_logic': StagesTransition.set_color_stage},
              {'name': 'LED2', 'transition_logic': StagesTransition.set_color_stage},
              {'name': 'LED3', 'transition_logic': StagesTransition.set_color_stage}]
    env = PlusMazeOneHotCues2ActiveDoors(stages=stages, relevant_cue=CueType.ODOR, stimuli_encoding=14)

    average_parameters, std_parameters = extract_model_average_fitting_parameters(fitting_file_name,
                                                                                  'MALearner.ACFTable')

    env_df = run_simulation_sampled_brain(env=env, brain_specs=[((TDBrain, MALearner, ACFTable),
                                                                 average_parameters, std_parameters,
                                                                 ([0.1, 10], [0.001, 0.4], [0.001, 0.4]))],
                                          repetitions=repetitions, initial_motivation=RewardType.NONE)

    return env_df


if __name__ == '__main__':
    reps = 20
    # fitting_file_name = 'fitting/Results/Rats-Results/Concatenated_Asymmetric_Fitting_Results.csv'
    # # all_simulation_data = run_increasing_IDShift(fitting_file_name, repetitions=reps)
    # all_simulation_data.to_csv(path_or_buf=f"/Users/georgekour/repositories/plus-maze-simulator/fitting/Results/simulations_results/increasing_ID_{reps}_{TRIALS_IN_DAY}TPD.csv", index=False)

    # all_models_simulation = run_dimensional_shifting_simulations_from_fitting_data(fitting_file_name, num_repetitions=reps)
    # all_models_simulation.to_csv(
    # 	path_or_buf=f"/Users/georgekour/repositories/plus-maze-simulator/fitting/Results/Rats-Results/identifiability_results/simulation_{reps}_{TRIALS_IN_DAY}TPD_loguniform_original_range_failing_allowed.csv",
    # 	index=False)

    all_simulation_data = run_dimensional_shifting_simulations(reps,
                                                               stages=PlusMazeOneHotCues2ActiveDoors.default_stages,
                                                               require_task_completion=True)

    all_simulation_data.to_csv(
        path_or_buf=f"/Users/georgekour/repositories/plus-maze-simulator/fitting/Results/Rats-Results/identifiability_results/simulation_{reps}_{TRIALS_IN_DAY}_nobias_symmetric.csv",
        index=False)

	# EDS_simulation_data = ED_shift_analysis(fitting_file_name, reps)
	# EDS_simulation_data.to_csv(
	# 	path_or_buf=f"/Users/georgekour/repositories/plus-maze-simulator/fitting/Results/Rats-Results/reported_results_dimensional_shifting/ED_shift_{reps}_{TRIALS_IN_DAY}TPD.csv",
	# 	index=False)
