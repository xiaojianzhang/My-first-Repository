#!/usr/bin/env python
# encoding: utf-8
r"""
Module containing all utility functions

:Author:
    Sheng Zhang -- Initial version (Aug 2016)
"""
import numpy
import multiprocessing
import scipy
import itertools 

season_year = "2015-2016"
X_LIM, Y_LIM = 47, 50
GRID_SIZE = 2
hoop_location = numpy.array([5.25, 25.0])

def compute_distance_square(Possessions):

	distance_square = 0
	num_datapoints = 0
	for possession in Possessions:
		mean_location_matrix = Possession.mean_location_matrix
		defense_locations = Possession.defensive_locations
		sampling_hidden_states = Possession.sampling_matchups_matrix
		N = mean_location_matrix.shape[1] #number of sequences in one possession
		num_datapoints += N * 5
		for seq in range(N):
			for def_num in range(5):
				matchup = sampling_hidden_states[seq,def_num]
				def_loc = defense_locations[def_num, seq, :]
				mean_location = mean_location_matrix[matchup,seq,:]
				distance_square += numpy.sum((def_loc - mean_location)**2)

	return distance_square, num_datapoints

def parallel_update_sigma_square(Possessions):

	Possessions_list = []
	length = len(Possessions)
	quotient, remainder = divmod(length, 40)
	for i in range(0, 39*quotient, quotient):
		Possessions_list.append(Possessions[i:i+quotient])
		
	Possessions_list.append(Possessions[39*quotient:])

	pool = multiprocessing.Pool(processes=40)
	results = [pool.apply_async(compute_distance_square,\
			  args=(Possessions,)) for Possessions in Possessions_list]

	pool.close()
	pool.join()
	output = [p.get() for p in results]
	total_distance_square = 0
	total_num_datapoints = 0
	for result in output:
		total_distance_square += result[0]
		total_num_datapoints += result[1]

	return total_distance_square, total_num_datapoints

def grid_location(player_location):

	if player_location.ndim == 1:
		
		#player_location, array, shape=(2, )
		loc_x = player_location[0] // GRID_SIZE
		loc_y = player_location[1] // GRID_SIZE
		
		if player_location[0] < 0:
			loc_x = 0
		if player_location[1] < 0:
			loc_y = 0
		if player_location[0] > (X_LIM-GRID_SIZE):
			loc_x = (X_LIM // GRID_SIZE) - 1   
		if player_location[1] > (Y_LIM-GRID_SIZE):
			loc_y = (Y_LIM // GRID_SIZE) - 1 
		
		return (int(loc_x), int(loc_y))

	else:
	#player_location, array, shape=(N, 2)
		grid_loc_list = []
		for player_loc in player_location:
			grid_loc_list.append(grid_location(player_loc))
	
	return grid_loc_list

def grid_num_x_y(X_LIM, Y_LIM, GRID_SIZE):
	
	num_grids_x = X_LIM // GRID_SIZE
	num_grids_y = Y_LIM // GRID_SIZE
	
	return num_grids_x, num_grids_y

def grid_cell_centers(X_LIM, Y_LIM, GRID_SIZE):

	num_grids_x, num_grids_y = grid_num_x_y(X_LIM, Y_LIM, GRID_SIZE)
	
	center_x = list(numpy.arange(GRID_SIZE/2.0, X_LIM, GRID_SIZE))
	if (num_grids_x * GRID_SIZE) < X_LIM:
		center_x[-1] = (X_LIM + (num_grids_x-1)*GRID_SIZE) / 2.0
		
	center_y = list(numpy.arange(GRID_SIZE/2.0, Y_LIM, GRID_SIZE))
	if (num_grids_y * GRID_SIZE) < Y_LIM:
		center_y[-1] = (Y_LIM + (num_grids_y-1)*GRID_SIZE) / 2.0
		
	centers_list = [i for i in itertools.product(center_x,center_y)]
	
	return centers_list

def mean_location_poss(off_locations, ball_locations, offense_Gammas):
	num_seq = ball_locations.shape[0]
	mean_location = numpy.zeros((5, num_seq, 2))

	if isinstance(offense_Gammas, list): 		#if offense_Gammas is a list, that means we are using player dependent Gammas
		if isinstance(offense_Gammas[0], dict): #if element is a dictionary, that means we are using grid dependent Gammas
			for i in range(num_seq): 		 	#for each sequence in that possession
				for j in range(5): 			 	#for each offensive player
					offense_location = off_locations[j,i,:]
					grid_loc = grid_location(offense_location)  	#grid_location at that moment e.g., (4,5)
					Gamma = offense_Gammas[j][grid_loc]     		#player_Gamma[(4,5)], array
					X = numpy.stack((offense_location, ball_locations[i], hoop_location), axis=1) #X=[2 by 3 matrix]
					mean_location[j,i,:] = numpy.dot(X, Gamma)

		elif isinstance(offense_Gammas[0], numpy.ndarray):  #if element is an 3 by 1 array, that means we are using one Gammas over court.
			for i in range(num_seq): 		 				#for each sequence in that possession
				for j in range(5): 			 				#for each offensive player
					offense_location = off_locations[j,i,:]
					Gamma = offense_Gammas[j]
					X = numpy.stack((offense_location, ball_locations[i], hoop_location), axis=1) #X=[2 by 3 matrix]
					mean_location[j,i,:] = numpy.dot(X, Gamma)

	elif isinstance(offense_Gammas, numpy.ndarray): #if offense_Gammas is an 3 by 1 array, that means we are using one Gammas over players and court.
		Gamma = offense_Gammas
		for i in range(num_seq): 		 			#for each sequence in that possession
			for j in range(5): 			 			#for each offensive player
				offense_location = off_locations[j,i,:]
				X = numpy.stack((offense_location, ball_locations[i], hoop_location), axis=1) #X=[2 by 3 matrix]
				mean_location[j,i,:] = numpy.dot(X, Gamma)

	return mean_location #matrix, shape=(5, num_seq, 2)

def likelihood_matrix_bonds(mean_location_matrix, def_locations, sigma_square):

	all_S = [list(i) for i in itertools.product(range(5), repeat=5)]
	num_seq = def_locations.shape[1]
	likelihood_matrix = numpy.ones((num_seq, 5**5))

	for i,S in enumerate(all_S):
		mean_location = numpy.zeros((5, num_seq, 2)) #matrix, shape=(5, num_seq, 2)
		for n in range(num_seq):
			for j,k in enumerate(S):
				mean_location[j,n,:] = mean_location_matrix[k,n,:]

		diff_mean_def_matrix = mean_location - def_locations #matrix, shape=(5, num_seq, 2)
		likelihood_matrix[:,i] = likelihood(diff_mean_def_matrix, sigma_square)		

	return likelihood_matrix #shape=[N, 5^5]

def likelihood_bonds(diff_mean_def_matrix, sigma_square):

	matrix = -numpy.sum(diff_mean_def_matrix**2.0,axis=2)/(2.0*sigma_square)

	likelihood_matrix = numpy.prod(numpy.exp(matrix)/(2.0*numpy.pi*sigma_square), axis=0)

	return likelihood_matrix #shape=[num_seq, 1]

def likelihood_matrix(mean_location_matrix, def_locations, sigma_square):

	num_seq = def_locations.shape[1]
	likelihood_matrix = numpy.ones((5, num_seq, 5))	#first dimension represents defensive player, last dimension means offensive player
	for j in range(5): #for each defensive player
		defense_location = def_locations[j,:,:] #shape=[num_seq, 2]
		for i in range(5): #for each offensive player
			mean_location = mean_location_matrix[i,:,:] #shape=[num_seq, 2]
			diff_mean_def_matrix = mean_location - defense_location #shape=[num_seq, 2]
			likelihood_matrix[j,:,i] = likelihood(diff_mean_def_matrix, sigma_square)

	return likelihood_matrix #shape=[5, num_seq, 5]

def likelihood(diff_mean_def_matrix, sigma_square):

	matrix = -numpy.sum(diff_mean_def_matrix**2.0,axis=1)/(2.0*sigma_square) #shape=[num_seq, 1]

	likelihood_matrix = numpy.exp(matrix)/(2.0*numpy.pi*sigma_square) #shape=[num_seq, 1]

	return likelihood_matrix #shape=[num_seq, 1]

def find_all_parent_matchups_index():

	parent_matchups_list=[]
	all_S = [list(i) for i in itertools.product(range(5), repeat=5)]
	for St in all_S:
		parent_matchups_index=[]
		all_St_plus_1 = all_possible_St_plus_1(St)
		for i in range(25):
			St_plus_1 = all_St_plus_1[i]		
			index = all_S.index(St_plus_1)
			parent_matchups_index.append(index)
		parent_matchups_list.append(parent_matchups_index)

	parent_matchups_index_list = parent_matchups_index

	return parent_matchups_list

def transition_probability_bonds(parent_matchups_list, energy_params):

	denominator_dict = parallel_St_Bt_plus_1_denominator(energy_params)
	all_S = [list(i) for i in itertools.product(range(5), repeat=5)]
	trans_prob_list = []
	for B in range(6):
		trans_prob_matrix = numpy.zeros((5**5, 25))
		for i,parent_matchups_index in enumerate(parent_matchups_list):
			St_plus_1 = all_S[i]
			St_plus_1_Bt_plus_1 = (St_plus_1, B)
			for index in parent_matchups_index:
				trans_prob=[]
				St = all_S[index] 
				St_Bt_plus_1 = (St, B)

				numerator_funval_grad, denominator_funval_grad = numerator_denominator_funvalue_gradient(energy_params, St_Bt_plus_1, St_plus_1_Bt_plus_1, denominator_dict)
				minus_logProb, Gradient = minus_logProb_Gradient_St_St_plus_1(numerator_funval_grad, denominator_funval_grad)	
				prob = numpy.exp(-minus_logProb)
				trans_prob.append(prob)
			trans_prob_matrix[i,:] = numpy.array(trans_prob)

		trans_prob_list.append(trans_prob_matrix)
	
	return trans_prob_list

def transition_matrix(rho):
		
		trans_matrix = numpy.ones([5,5]) * ((1-rho)/4.0)
		numpy.fill_diagonal(trans_matrix, rho)
		
		return trans_matrix

def sampling_from_posterior_dist_bonds(likelihood_matrix, alpha, energy_params, ball_states, trans_prob_list):

	all_S = [list(i) for i in itertools.product(range(5), repeat=5)]
	num_seq = likelihood_matrix.shape[0]

	#backward_sampling:
	#initialization:
	prob_mass = alpha[-1,:] / numpy.sum(alpha[-1,:])
	sample = numpy.random.choice(5**5, 1, p=list(prob_mass))[0]
	sample_list=[all_S[sample]]
	
	#iterations:
	for i in range(num_seq - 2, -1, -1):
		transition_matrix = trans_prob_list[ball_states[i]]
		St_index = [l for l in parent_matchups_list[sample]]
		alpha_array = numpy.array([alpha[i,l] for l in parent_matchups_list[sample]])
		denominator = numpy.dot(transition_matrix[sample,:], alpha_array)
		numerator = transition_matrix[sample,:] * alpha_array
		prob_mass = numerator / denominator
		sample = numpy.random.choice(25, 1, p=list(prob_mass))[0]
		sample = St_index[sample]
		sample_list.insert(0, all_S[sample])
		sample_array = numpy.asarray(sample_list)
	
	return sample_array

def sampling_from_posterior_dist(likelihood_matrix, alpha, transition_matrix):

	num_seq = likelihood_matrix.shape[0]

	#backward_sampling:
	#initialization:
	prob_mass = alpha[-1,:] / numpy.sum(alpha[-1,:])
	sample = numpy.random.choice(5, 1, p=list(prob_mass))[0]
	sample_list=[sample]
	
	#iterations:
	for i in range(num_seq - 2, -1, -1):
		denominator = numpy.sum(transition_matrix[:,sample] * alpha[i, :])
		numerator = transition_matrix[:,sample] * alpha[i, :]
		prob_mass = numerator / denominator
		sample = numpy.random.choice(5, 1, p=list(prob_mass))[0]
		sample_list.insert(0, sample)

	return sample_list

def log_llh_given_matchups_bonds(sampling_matchups_list, likelihood_matrix):
	
	log_llh_given_I = 0
	N = len(sampling_matchups_list)

	all_S = [list(i) for i in itertools.product(range(5), repeat=5)]

	for i in range(N):
		sample = sampling_matchups_list[i]
		index = all_S.index(sample)
		log_llh_given_I += numpy.log(likelihood_matrix[i, index])

	return log_llh_given_I

def log_llh_given_matchups(sampling_matchups_matrix, likelihood_matrix):
	
	log_llh_given_I = 0
	N = sampling_matchups_matrix.shape[0]

	for num_seq in range(N):
		matchup = sampling_matchups_matrix[num_seq,:]
		for j in range(5):
			offense = matchup[j]
			defense = j
			log_llh_given_I += numpy.log(likelihood_matrix[defense,num_seq,offense])

	return log_llh_given_I

def rho_update(Possessions):

	summation1 = 0.0
	summation2 = 0.0
	for i in range(len(Possessions)):
		for j in range(5):
			for k in range(5):
				summation1 = summation1 + sum([sum(Possessions[i].E_two_hidden_states_product_list[j][:,k,kk]) for kk in range(5) if kk == k])
				summation2 = summation2 + sum([sum(Possessions[i].E_two_hidden_states_product_list[j][:,k,kk]) for kk in range(5) if kk != k])


	#update parameter rho
	Q = 0.25 * summation1/summation2
	rho_new = Q/(1+Q)

	return rho_new

if __name__ == "__main__":
	pass





