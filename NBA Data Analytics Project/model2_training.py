#!/usr/bin/env python
# encoding: utf-8
r"""
Script containing function for training model: (1) grid, player dependent Gammas (2) using rho as transition parameter. 

:Author:
    Sheng Zhang -- Initial version (Sep 2016)
"""
import numpy
import os
import pickle

from sampling_Gammas import get_F_T_projection, d1_square_d2_square, kernel_regression_estimate, Calculate_W_V, update_Gamma_sampling
from class_objects import Player, Shared_Parameter
from utils import data_partition, count_datpoint, transition_matrix
from E_step import parallel_E_step

if __name__ == "__main__":

	#pdb.set_trace()

	iteration = 1
	f = open("common_game_poss_dict.pickle",'rb')
	common_game_poss_dict = pickle.load(f)
	
	team_list = ['GS', 'SA', 'OKC', 'Cle', 'LAC', 'Mia']
	player_list = []
	for team in team_list:
		player_list += common_game_poss_dict[team][1]

	sampling_parameters = {'nu_square': [10.0, 10.0, 10.0], 'l_square' : [(0.2, 0.2), (0.2, 0.2), (0.2, 0.2)]} 
	
	PROJ_PERP_MATRIX, V_BAR = get_F_T_projection()
	V_BAR = numpy.matrix(V_BAR).T
	
	d1_d2_list = d1_square_d2_square(8.0)

	Possessions=[]
	for data_file in os.listdir("./input_data/"):
		if data_file.split('_')[0] in team_list:
			f = open("./input_data/"+data_file, 'rb')
			Possessions += pickle.load(f)
			print(data_file)

	Possessions_training, Possessions_testing = data_partition(Possessions, 0.20)
	datapoints_training = count_datpoint(Possessions_training)
	datapoints_testing = count_datpoint(Possessions_testing)
	Players={}
	for player in player_list:
		Players[player[0]] = Player(player[0], player[1], None, True)

	global_parameter = Shared_Parameter()
	global_parameter_list=[]
	total_log_likelihood_list=[]

	trans_matrix = transition_matrix(global_parameter.rho)
	Possessions_training, total_log_likelihood_training, total_log_llh_given_I_training = parallel_E_step(Possessions_training, Players, global_parameter, trans_matrix)
	Possessions_testing, total_log_likelihood_testing, total_log_llh_given_I_testing = parallel_E_step(Possessions_testing, Players, global_parameter, trans_matrix)
	print('total_log_likelihood_test = {}'.format(total_log_likelihood_testing))
	print('total_log_likelihood_training = {}'.format(total_log_likelihood_training))
	total_log_likelihood_list.append((iteration, total_log_likelihood_training, total_log_likelihood_testing, \
									  total_log_likelihood_training/datapoints_training, total_log_likelihood_testing/datapoints_testing, \
									  total_log_llh_given_I_training/datapoints_training, total_log_llh_given_I_testing/datapoints_testing))
	file = open("./output_results/log_likelihood_list.pickle", 'wb')
	pickle.dump(total_log_likelihood_list, file)
	print('log_likelihood_list dumped in ./output_results')

	global_parameter.update_rho(Possessions_training)
	global_parameter.update_sigma_square_sampling(Possessions_training)
	print('rho = {}'.format(global_parameter.rho))
	print('sigma_square = {}'.format(global_parameter.sigma_square))
	global_parameter_list.append((iteration, global_parameter.rho, global_parameter.sigma_square))
	file = open("./output_results/global_parameter_list.pickle", 'wb')
	pickle.dump(global_parameter_list, file)
	print('global_parameter_list dumped in ./output_results')

	Players = Calculate_W_V(Possessions_training, Players)
	Players, mean_Gammas = update_Gamma_sampling(Players, sampling_parameters, PROJ_PERP_MATRIX, V_BAR, global_parameter.sigma_square)
	file = open("./output_results/Players.pickle", 'wb')
	pickle.dump(Players, file)
	print('Players.pickle dumped in ./output_results')
	file = open("./output_results/mean_Gammas.pickle", 'wb')
	pickle.dump(mean_Gammas, file)
	print('mean_Gammas.pickle dumped in ./output_results')
	print('iteration{} completed'.format(iteration))

	while(True):
		iteration += 1
		trans_matrix = transition_matrix(global_parameter.rho)
		Possessions_training, total_log_likelihood_training, total_log_llh_given_I_training = parallel_E_step(Possessions_training, Players, global_parameter, trans_matrix)
		Possessions_testing, total_log_likelihood_testing, total_log_llh_given_I_testing = parallel_E_step(Possessions_testing, Players, global_parameter, trans_matrix)
		print('total_log_likelihood_testing = {}'.format(total_log_likelihood_testing))
		print('total_log_likelihood_training = {}'.format(total_log_likelihood_training))
		total_log_likelihood_list.append((iteration, total_log_likelihood_training, total_log_likelihood_testing, \
									  	  total_log_likelihood_training/datapoints_training, total_log_likelihood_testing/datapoints_testing, \
									      total_log_llh_given_I_training/datapoints_training, total_log_llh_given_I_testing/datapoints_testing))
		file = open("./output_results/log_likelihood_list.pickle", 'wb')
		pickle.dump(total_log_likelihood_list, file)
		print('log_likelihood_list dumped in ./output_results')
		
		sampling_parameters = kernel_regression_estimate(Players, d1_d2_list)
		print('sampling_parameters = {}'.format(sampling_parameters))
		global_parameter.update_rho(Possessions_training)
		global_parameter.update_sigma_square_sampling(Possessions_training)
		print('energy_params = {}'.format(global_parameter.rho))
		print('sigma_square = {}'.format(global_parameter.sigma_square))
		global_parameter_list.append((iteration, global_parameter.rho, global_parameter.sigma_square))
		file = open("./output_results/global_parameter_list.pickle", 'wb')
		pickle.dump(global_parameter_list, file)
		print('global_parameter_list dumped in ./output_results')

		Players = Calculate_W_V(Possessions_training, Players)

		Players, mean_Gammas = update_Gamma_sampling(Players, sampling_parameters, PROJ_PERP_MATRIX, V_BAR, global_parameter.sigma_square)
		file = open("./output_results/Players.pickle", 'wb')
		pickle.dump(Players, file)
		print('Players.pickle dumped in ./output_results')
		file = open("./output_results/mean_Gammas.pickle", 'wb')
		pickle.dump(mean_Gammas, file)
		print('mean_Gammas.pickle dumped in ./output_results')
		print('iteration{} completed'.format(iteration))










