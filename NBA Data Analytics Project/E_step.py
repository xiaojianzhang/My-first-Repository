#!/usr/bin/env python
# encoding: utf-8
r"""
Module containing all functions for E_step

:Author:
    Sheng Zhang -- Initial version (Aug 2016)
"""

import numpy
import itertools
import multiprocessing
import scipy

from energy_bonds import state_energy_representation
from utils import sampling_from_posterior_dist, log_llh_given_matchups, mean_location_poss, likelihood_matrix, \
				  sampling_from_posterior_dist_bonds, log_llh_given_matchups_bonds, likelihood_matrix_bonds

def init_matchup_dist_bonds(energy_params, B0):
	#energy_params = [E_onball_open, E_offball_1on1, E_onball_2on1, E_offball_2on1, Trans_cost]
	params = numpy.array(energy_params)
	all_S0 = [list(i) for i in itertools.product(range(5), repeat=5)]
	energy_list = []
	for S0 in all_S0:
		S0_B0 = (S0, B0)
		energy_coeff = state_energy_representation(S0_B0)
		energy = numpy.exp(-numpy.dot(energy_coeff, params[:4]))
		energy_list.append(energy)

	S0_prob_dist = numpy.array(energy_list) / sum(energy_list)	

	return S0_prob_dist #array

def init_matchup_dist(E_hidden_state_list):
	if E_hidden_state_list == None:
		return numpy.ones([5,5]) * 0.2
	
	else:  #update init_matchup_dist
		init_hidden_state_dist = numpy.zeros([5,5])
		
		for i,E_hidden_state in enumerate(E_hidden_state_list):
			init_hidden_state_dist[i,:] = E_hidden_state[0,:] / numpy.sum(E_hidden_state[0,:])
			
		return init_hidden_state_dist

def forward_backward_rescaled_sampling(likelihood_matrix, initial_hidden_state_dist, transition_matrix):

	num_seq = likelihood_matrix.shape[1] #number of sequences in a possession
	sampling_matchups_list = []
	E_hidden_state_list = []
	E_two_hidden_states_product_list =	[]
	log_likelihood = 0

	for def_num in range(5):#for each defensive player
		#initialization:
		alpha = numpy.zeros([num_seq,5])
		alpha_rescaled = numpy.zeros([num_seq,5])
		beta_rescaled = numpy.zeros([num_seq,5])
		c = []
		for k in range(5):
			alpha[0,k] = initial_hidden_state_dist[def_num, k] * likelihood_matrix[def_num,0,k]
		c.append(sum(alpha[0,:]))
		alpha_rescaled[0,:] = alpha[0,:] / c[0]
		beta_rescaled[-1,:] = numpy.ones(5)

		#iterations:
		for n in range(1,num_seq):
			for k in range(5):
				alpha[n,k] = likelihood_matrix[def_num,n,k] * numpy.dot(alpha_rescaled[n-1,:], transition_matrix[k,:])
			c.append(sum(alpha[n,:]))
			alpha_rescaled[n,:] = alpha[n,:] / c[n]

		for n in range(num_seq-2,-1,-1):
			for k in range(5):
				beta_rescaled[n,k] = sum(beta_rescaled[n+1,:] * likelihood_matrix[def_num,n+1,:] * transition_matrix[k,:]) / c[n+1]
		
		#Evaluations:
		log_likelihood += sum(numpy.log(c))  #log(P(D_j1, D_j2,...,D_jN| parameters))
		E_hidden_state = alpha_rescaled * beta_rescaled

		E_two_hidden_states_product = numpy.empty([num_seq-1,5,5])

		for n in range(1,num_seq,1):
			for k in range(5):#index for I_n
				for kk in range(5):#index for I_(n-1)
					E_two_hidden_states_product[n-1, k, kk] = c[n] * alpha_rescaled[n-1,kk] * likelihood_matrix[def_num,n,k] * transition_matrix[k,kk] * beta_rescaled[n,k]

		sampling_matchups_list.append(sampling_from_posterior_dist(likelihood_matrix[def_num,:,:], alpha, transition_matrix))
		E_hidden_state_list.append(E_hidden_state)
		E_two_hidden_states_product_list.append(E_two_hidden_states_product)

	sampling_matchups_matrix = numpy.asarray(sampling_matchups_list, dtype=numpy.int).T #shape=[5,num_seq]
	log_llh_given_I = log_llh_given_matchups(sampling_matchups_matrix, likelihood_matrix) #log(P(D_j1, D_j2,...,D_jN| parameters, I))

	return E_hidden_state_list, E_two_hidden_states_product_list, sampling_matchups_matrix, log_likelihood, log_llh_given_I

def E_step(Possessions, Players, global_parameter, transition_matrix):
	total_log_likelihood=0
	total_log_likelihood_given_I=0
	for k in range(len(Possessions)):
		if k % 50 == 0 and k != 0:
			print("{} possessions have been finished".format(k))

		offense_locations = Possessions[k].offensive_locations #shape=[5,num_seq,2]
		defense_locations = Possessions[k].defensive_locations #shape=[5,num_seq,2]
		ball_locations = Possessions[k].ball_location          #shape=[num_seq,2]
		offensive_players = Possessions[k].offensive_players   #shape=[5,]
		
		
		offense_Gammas = [Players[offensive_players[i]].Gammas for i in range(5)]
		mean_location_matrix = mean_location_poss(offense_locations, ball_locations, offense_Gammas) #matrix, shape=(5, num_seq, 2)

		likelihood_matrix = likelihood_matrix(mean_location_matrix, defense_locations, global_parameter.sigma_square) #shape=[5, num_seq, 5]
		Possessions[k].mean_location_matrix = mean_location_matrix #matrix, shape=(5, num_seq, 2)
		
		init_hidden_state_dist = init_matchup_dist(Possessions[k].E_hidden_state_list)
		E_hidden_state_list, E_two_hidden_states_product_list, sampling_matchups_matrix, log_likelihood, log_llh_given_I \
			= forward_backward_rescaled_sampling(likelihood_matrix, init_hidden_state_dist, transition_matrix)
		
		Possessions[k].sampling_matchups_matrix = sampling_matchups_matrix
		Possessions[k].E_hidden_state_list = E_hidden_state_list
		Possessions[k].E_two_hidden_states_product_list = E_two_hidden_states_product_list
				
		total_log_likelihood += log_likelihood	
		total_log_likelihood_given_I += log_llh_given_I

	return Possessions, total_log_likelihood, total_log_likelihood_given_I

def parallel_E_step(Possessions, Players, global_parameter, transition_matrix):
	
	Possessions_list = []
	length = len(Possessions)
	quotient, remainder = divmod(length, 40)
	for i in range(0, 39*quotient, quotient):
		Possessions_list.append(Possessions[i:i+quotient])
		
	Possessions_list.append(Possessions[39*quotient:])

	pool = multiprocessing.Pool(processes=40)
	results = [pool.apply_async(E_step,\
			  args=(Possessions, Players, global_parameter, transition_matrix)) for Possessions in Possessions_list]

	pool.close()
	pool.join()
	output = [p.get() for p in results]
	total_log_likelihood = 0
	total_log_likelihood_given_I = 0
	Possessions_new = []
	for i,result in enumerate(output):
		Possessions_new += result[0]
		total_log_likelihood += result[1]
		total_log_likelihood_given_I += result[2]

	print("E_step has been done!!!")

	return Possessions_new, total_log_likelihood, total_log_likelihood_given_I


def forward_smoothing_backward_sampling_bonds(likelihood_matrix, initial_state_dist, energy_params, ball_states, trans_prob_list):

	num_seq = likelihood_matrix.shape[0]
	#initialization:
	alpha = numpy.zeros((num_seq, 5**5))
	alpha_rescaled = numpy.zeros((num_seq, 5**5))
	c = []
	alpha[0,:] = initial_state_dist * likelihood_matrix[0,:]
	c.append(sum(alpha[0,:]))
	alpha_rescaled[0,:] = alpha[0,:] / c[0]
	
	#iterations:
	for n in range(1,num_seq):
		transition_matrix = trans_prob_list[ball_states[n]]
		for k in range(5**5):
			trans_prob_array = transition_matrix[k,:]
			alpha_rescaled_array = numpy.array([alpha_rescaled[n-1,l] for l in parent_matchups_list[k]])
			alpha[n,k] = likelihood_matrix[n,k] * numpy.dot(alpha_rescaled_array, trans_prob_array)
		c.append(sum(alpha[n,:]))
		alpha_rescaled[n,:] = alpha[n,:] / c[n]

	#Evaluations:
	log_likelihood = sum(numpy.log(c))  #log(P(D_j1, D_j2,...,D_jN| parameters))

	#Sampling
	sampling_matchups_list = sampling_from_posterior_dist_bonds(likelihood_matrix, alpha, energy_params, ball_states, trans_prob_list)
	log_llh_given_I = log_llh_given_matchups_bonds(sampling_matchups_list, likelihood_matrix) #log(P(D_j1, D_j2,...,D_jN| parameters, I))
	sampling_matchups_matrix = numpy.asarray(sampling_matchups_list, dtype=numpy.int)

	return sampling_matchups_matrix, log_likelihood, log_llh_given_I


def E_step_bonds(Possessions, Players, global_parameter,trans_prob_list):

	total_log_likelihood=0
	total_log_likelihood_given_I=0
	for k in range(len(Possessions)):
		if k % 50 == 0 and k != 0:
			print("{} possessions finished".format(k))

		offense_locations = Possessions[k].offensive_locations
		defense_locations = Possessions[k].defensive_locations
		ball_locations = Possessions[k].ball_location
		offensive_players = Possessions[k].offensive_players
		ball_states = Possessions[k].ball_states

		initial_state_dist = init_matchup_dist_bonds(global_parameter.energy_params, ball_states[0])

		offense_Gammas = [Players[offensive_players[i]].Gammas for i in range(5)]
		mean_location_matrix = mean_location_poss(off_location, ball_location, offense_Gammas) #shape=(5, N, 2)
		Possessions[k].mean_location_matrix = mean_location_matrix
		
		likelihood_matrix = likelihood_matrix_bonds(mean_location_matrix, defense_locations, global_parameter.sigma_square) #shape=(N, 5**5)
		
		sampling_matchups_matrix, log_likelihood, log_llh_given_I = \
			forward_smoothing_backward_sampling_bonds(likelihood_matrix, initial_state_dist, global_parameter.energy_params, ball_states, trans_prob_list)
		
		Possessions[k].sampling_matchups_matrix = sampling_matchups_matrix

		total_log_likelihood += log_likelihood
		total_log_likelihood_given_I += log_llh_given_I

	print("this process has been finished")

	return Possessions, total_log_likelihood, total_log_likelihood_given_I

def parallel_E_step_bonds(Possessions, Players, global_parameter, trans_prob_list):
	
	Possessions_list = []
	length = len(Possessions)
	quotient, remainder = divmod(length, 40)
	for i in range(0, 39*quotient, quotient):
		Possessions_list.append(Possessions[i:i+quotient])
		
	Possessions_list.append(Possessions[39*quotient:])

	pool = multiprocessing.Pool(processes=40)
	results = [pool.apply_async(E_step_bonds,\
			  args=(Possessions, Players, global_parameter, trans_prob_list)) for Possessions in Possessions_list]

	pool.close()
	pool.join()
	output = [p.get() for p in results]
	total_log_likelihood = 0
	total_log_likelihood_given_I = 0
	Possessions_new = []
	for i,result in enumerate(output):
		Possessions_new += result[0]
		total_log_likelihood += result[1]
		total_log_likelihood_given_I += result[2]

	print("E_step has been done!!!")

	return Possessions_new, total_log_likelihood, total_log_likelihood_given_I


if __name__ == "__main__":
	pass