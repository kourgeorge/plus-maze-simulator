
from fitting.MazeResultsBehaviouralLC import *


if __name__ == '__main__':

	#file_path = '/Users/gkour/repositories/plusmaze/fitting/Results/Rats-Results/fitting_results_2023_01_31_18_29_50_tmp.csv'
	file_path = '/Users/gkour/repositories/plusmaze/fitting/Results/Rats-Results/fitting_results_motivation_reported_30_1_new.csv'

	####################################################
	####################################################
	
	#Fig 2: Animals Choice Accuracy, preference, and days to criterion.
	# learning_curve_behavioral_boxplot(file_path)
	# water_food_correct(file_path, 'water')
	# water_food_correct(file_path, 'food')
	# show_days_to_criterion(file_path)
	# goal_choice_index(file_path)
	# show_fitting_parameters(file_path)

	####################################################
	####################################################
	
	#Fig 3: Action biases depend on the motivational context
	# plot_models_fitting_result_per_stage_action_bias(file_path)
	# models = utils.flatten_list([[m,'B-'+m, 'M(B)-'+m] for m in ['SARL','ORL','FRL']])
	# compare_fitting_criteria(file_path, models=models)
	# average_likelihood_animal(file_path, models=models)


	# Additional results: Success rate.
	compare_model_subject_learning_curve_average(file_path, ['M(B)-'+m for m in ['SARL','ORL','FRL']])
	compare_model_subject_learning_curve_average(file_path, ['B-' + m for m in ['SARL', 'ORL', 'FRL']])
	compare_model_subject_learning_curve_average(file_path, ['SARL', 'ORL', 'FRL'])
	# show_fitting_parameters(file_path)

	# Fig 4: Separate reinforcement learning systems for different motivational contexts
	# models = utils.flatten_list([['M(B)-' + m, 'M(V)-' + m, 'M(VB)-' + m] for m in ['SARL', 'ORL', 'FRL']])
	# pairs = utils.flatten_list(
	# 	[[((m, 'M(B)-m'), (m, 'M(V)-m')), 
	# 	  ((m, 'M(B)-m'), (m, 'M(VB)-m')), 
	# 	  ((m, 'M(V)-m'), (m, 'M(VB)-m'))] for m in
	# 	 ['SARL', 'ORL', 'FRL']])
	# 
	# compare_fitting_criteria(file_path, models=models)
	# average_likelihood_simple(file_path, models=models,  pairs=pairs)
	# models_fitting_quality_over_times_average(file_path, models=utils.flatten_list([['M(B)-' + m, 'M(V)-' + m, 'M(VB)-' + m] for m in ['SARL']]))


	x=1

	#Fig 5: observation-action dependency on environmental context.

	# models = ['FRL','E(V)-FRL', 'E(V)-M(B)-FRL', 'E(V)-M(VB)-FRL' ]
	# pairs = [ (('FRL', 'm'), ('FRL','E(V)-m')), (('FRL', 'E(V)-m'), ('FRL', 'E(V)-M(B)-m')),
	# 								 (('FRL', 'E(V)-M(VB)-m'), ('FRL', 'E(V)-M(B)-m')),
	# 		 							(('FRL', 'E(V)-m'), ('FRL', 'E(V)-M(VB)-m'))
	# 		,(('FRL', 'm'), ('FRL','E(V)-M(B)-m')),(('FRL', 'm'), ('FRL','E(V)-M(VB)-m'))]
	# 
	# average_likelihood_simple(file_path, models, pairs)
	# compare_fitting_criteria(file_path, models)
	# models_fitting_quality_over_times_average(file_path, models=models)


	
	# New experiment - B(a) independant of Environmental context
	# models = ['SARL', 'E(B)-SARL','ORL', 'E(B)-ORL', 'FRL','E(B)-FRL'] #'E(B)-M(B)-FRL', 'E(V)-M(B)-FRL', 'E(VB)-M(B)-FRL' ]
	# pairs = [ (('SARL', 'm'), ('SARL','E(B)-m')), (('ORL', 'm'), ('ORL','E(B)-m')), (('FRL', 'm'), ('FRL','E(B)-m'))]
	# 
	# compare_fitting_criteria(file_path, models, pallete='Paired')
	# average_likelihood_simple(file_path, models, pairs)
	# models_fitting_quality_over_times_average(file_path, models=models, palette='Paired')
	# 
	

	#Figure 7: Parameters analysis
	# show_fitting_parameters(file_path)
	average_nmr_animal(file_path)

	# file_path = 'fitting/Results/Rats-Results/fitting_results_nmr_M(B).csv'
	# compare_model_subject_learning_curve_average(file_path)
	# show_fitting_parameters(file_path)
	# bias_effect_nmr(file_path)

	# average_likelihood_simple(file_path, models=utils.flatten_list([['M(B)-'+m] for m in ['SARL','ORL','FRL']]))
	# models_fitting_quality_over_times_average(file_path, models=utils.flatten_list([['M(B)-'+m] for m in ['SARL','ORL','FRL']]))

	####################################################
	####################################################

	# Fig 6: Action bias response to changes.
	models = ['B-' + m for m in ['SARL', 'ORL', 'FRL']]
	# bias_variables_in_stage(file_path, models)
	# model_values_development(file_path, models)

	models = ['M(B)-' + m for m in ['SARL', 'ORL']]+['E(V)-M(B)-FRL']
	bias_variables_in_stage(file_path, models)
	model_values_development(file_path, models)
	# show_fitting_parameters(file_path)


