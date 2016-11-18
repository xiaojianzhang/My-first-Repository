#!/usr/bin/env python
# encoding: utf-8
r"""
Module containing functions for sampling Gammas

:Author:
    Sheng Zhang -- Initial version (Aug 2016)
    Suraj Keshiri
"""
import numpy
import multiprocessing
import scipy
from scikits.sparse import cholmod
import itertools
import random
from utils import grid_location, grid_num_x_y, grid_cell_centers, find_player_possessions

X_LIM, Y_LIM = 47, 50
GRID_SIZE = 2
hoop_location = numpy.array([5.25, 25.0])

def Calculate_W_V(Possessions, Players):

	player_possessions_dict = find_player_possessions(Possessions)
	for player_id in Players.keys():
		poss_index_list = player_possessions_dict[player_id]
		num_grids_x, num_grids_y = grid_num_x_y(X_LIM, Y_LIM, GRID_SIZE)
		W = {}
		V = {}
		num_grids = {}

		for i in range(num_grids_x):
			for j in range(num_grids_y):
				W[(i,j)] = numpy.zeros((3,3))
				V[(i,j)] = numpy.zeros(3)
				num_grids[(i,j)] = 0
				

		for k in poss_index_list:
			possession = Possessions[k]
			index = possession.offensive_players.index(player_id)
			off_location = possession.offensive_locations[index]
			ball_location = possession.ball_location
			matchups_matrix = possession.sampling_matchups_matrix
			def_location = possession.defensive_locations
			num_seq = ball_location.shape[0]
			for i in range(num_seq):
				grid_loc = grid_location(off_location[i])
				Z = numpy.vstack((off_location[i], ball_location[i], hoop_location))

				for j in range(5):
					if matchups_matrix[i,j] == index:
						W[grid_loc] += numpy.dot(Z, Z.T)
						d = def_location[j, i, :]
						V[grid_loc] += numpy.dot(Z, d)
						num_grids[grid_loc] += 1

		Players[player_id].W = W
		Players[player_id].V = V
		Players[player_id].num_grid_datapoints = num_grids
		
	
	return Players

def update_player_Gamma(Player, Gammas_sample):
	Gammas_dict = Player.Gammas
	GRID_CENTERS = grid_cell_centers(X_LIM, Y_LIM, GRID_SIZE)
	N = len(GRID_CENTERS)
	num_grids_x, num_grids_y = grid_num_x_y(X_LIM, Y_LIM, GRID_SIZE)
	Gammas = Gammas_sample.reshape((N,3))
	row = 0
	for i in range(num_grids_x):
		for j in range(num_grids_y):
			Gammas_dict[(i,j)] = Gammas[row,:]
			row += 1

	return Gammas_dict

def covariance_calculation(samples_matrix, d1_d2_list):

	covariance_list = []
	covariance_matrix = numpy.cov(samples_matrix)
	N = covariance_matrix.shape[0]
	for ele in d1_d2_list:
		i = ele[0]
		j = ele[1]
		covariance_list.append(covariance_matrix[i,j])

	covariance = numpy.array(covariance_list)

	return covariance

def d1_square_d2_square(threshold):

	d1_square = []
	d2_square = []
	GRID_CENTERS = grid_cell_centers(X_LIM, Y_LIM, GRID_SIZE)
	d1_d2_list = []
	N = len(GRID_CENTERS)
	for i in range(N):
		for j in range(i,N):
			center_i = GRID_CENTERS[i]
			center_j = GRID_CENTERS[j]
			d1_square = (center_i[0]-center_j[0])**2.0 
			d2_square = (center_i[1]-center_j[1])**2.0
			if numpy.sqrt(d1_square+d2_square) < threshold:
				d1_d2_list.append([i, j, d1_square, d2_square])

	return d1_d2_list

def simple_regression(X, Y):

	parameter = numpy.linalg.solve(numpy.dot(X.T, X), numpy.dot(X.T, Y))

	parameter[0] = numpy.exp(parameter[0])
	parameter[1] = 1.0 / parameter[1]
	parameter[2] = 1.0 / parameter[2]

	parameter = [parameter[0], (parameter[1], parameter[2])]

	return parameter

def kernel_regression_estimate(Players, d1_d2_list):

	num_grids_x, num_grids_y = grid_num_x_y(X_LIM, Y_LIM, GRID_SIZE)
	samples_dict = {'off':{}, "ball": {}, "hoop":{}}
	N = len(Players.keys())
	total_grids = num_grids_x * num_grids_y

	samples_dict['off'] = numpy.zeros((total_grids, N))
	samples_dict['ball'] = numpy.zeros((total_grids, N))
	samples_dict['hoop'] = numpy.zeros((total_grids, N))
	
	row = 0
	for i in range(num_grids_x):
		for j in range(num_grids_y):
			for k,player_id in enumerate(Players.keys()):
				Gamma = Players[player_id].Gammas[(i,j)]
				samples_dict['off'][row,k]=Gamma[0]
				samples_dict['ball'][row,k]=Gamma[1]
				samples_dict['hoop'][row,k]=Gamma[2]
			row += 1

	off_cov = covariance_calculation(samples_dict['off'], d1_d2_list)
	ball_cov = covariance_calculation(samples_dict['ball'], d1_d2_list)
	hoop_cov = covariance_calculation(samples_dict['hoop'], d1_d2_list)
	

	num_rows = len(d1_d2_list)
	d1_square = numpy.array([ele[2] for ele in d1_d2_list])
	d2_square = numpy.array([ele[3] for ele in d1_d2_list])
	x1 = numpy.ones(num_rows)
	x2 = -0.5 * numpy.array(d1_square) 
	x3 = -0.5 * numpy.array(d2_square)

	X = numpy.vstack((x1, x2, x3)).T
	prarameters = {"off": None, "ball": None, "hoop": None} 
	prarameters['off'] = simple_regression(X, off_cov)
	prarameters['ball'] = simple_regression(X, ball_cov)
	prarameters['hoop'] = simple_regression(X, hoop_cov)
	

	nu_square = [prarameters['off'][0], prarameters['ball'][0], prarameters['hoop'][0]]
	l_square = [prarameters['off'][1], prarameters['ball'][1], prarameters['hoop'][1]]
	parameters= {'nu_square': nu_square, 'l_square': l_square}
	
	return parameters 


def kernel(nv_sq, l_sq, x, y):
	""" Defines the gaussian kernel function"""
	edist_square = sum(pow(i-j,2)/l for i, j, l in zip(x,y,l_sq))
	return nv_sq*numpy.exp(-0.5*edist_square)

def cov_matrix(nv_sq, l_sq):
	""" OUTPUT: Generates the covariance matrix for the GRID_CENTERS on the court
		INPUT: nv_sq and l_sq corresponds to the parameters of the kernel
	"""
	GRID_CENTERS = grid_cell_centers(X_LIM, Y_LIM, GRID_SIZE)
	# Initialize the covariance matrix with nans
	cov_matrix = numpy.empty((len(GRID_CENTERS), len(GRID_CENTERS)))
	cov_matrix[:] = numpy.NAN
	# fill up the matrix with covariances (not repeating because of symmetry)
	for a, b in itertools.product(
				enumerate(GRID_CENTERS), enumerate(GRID_CENTERS)): 
		if numpy.isnan(cov_matrix[a[0],b[0]]):
			cov_value = kernel(nv_sq, l_sq, a[1], b[1])
			cov_matrix[a[0],b[0]], cov_matrix[b[0],a[0]] = cov_value, cov_value
	return cov_matrix

def full_cov_matrix_inverse(nu_sq_list, l_sq_list):
	""" 
	INPUT: list of nv_sq and l_sq that corresponds to each gamma element. 
	OUTPUT: Inverse covariance of stacked gamma vector in csr_matrix format

	Generates the inverse of covariance matrix of the stacked Gamma vector 
	over all the grids. Since each element of Gamma is independent, we can 
	invert the covariance of each element over the court independently. 
	Then to we stack the each gamma value vector over the court and permute 
	them to get the stacked Gamma vector. We can use the covariance of 
	each element of gamma to calculate the final covariance matrix
	"""
	import scipy.sparse as sp
	GRID_CENTERS = grid_cell_centers(X_LIM, Y_LIM, GRID_SIZE)
	N = len(GRID_CENTERS)
	cov_matrix_list = [cov_matrix(nv_sq, l_sq)
						 for nv_sq, l_sq in zip(nu_sq_list, l_sq_list)]
	inv_cov_matrix_list = [numpy.linalg.inv(A+0.0001*numpy.eye(N)) for A in cov_matrix_list]
	
	# Permutation matrix
	P = sp.lil_matrix((3*N, 3*N))
	for i in range(3*N):
		P[i,N*(i%3)+(i//3)] = 1
	P = P.tocsr()
	
	# stack the individual covariance matrices
	full_inv_cov = sp.lil_matrix((3*N,3*N))
	full_inv_cov[0:N,0:N] = inv_cov_matrix_list[0]
	full_inv_cov[N:2*N,N:2*N] = inv_cov_matrix_list[1]
	full_inv_cov[2*N:3*N,2*N:3*N] = inv_cov_matrix_list[2]
	full_inv_cov = full_inv_cov.tocsr()

	# get the final covariance matrix of Gamma vector
	full_inv_cov = P*full_inv_cov*P.T

	return full_inv_cov

def get_posterior_params(W, V, M, K_inv, sigma_square):
	"""
	INPUT: 	W, sparse matrix format, same dimension as K_inv
			V, a numpy vector of size K_inv.shape[0]
			M: the mean vector in numpy format
			K_inv: precision matrix in sparse form
	"""
	from scikits.sparse import cholmod
	posterior_precision = K_inv + W/sigma_square
	posterior_precision_factor = cholmod.cholesky(posterior_precision)
	posterior_mean = posterior_precision_factor(V/sigma_square + K_inv*M)
	return (posterior_mean, posterior_precision) 

def get_F_T_projection():
	"""
	INPUT: Takes 
	OUTPUT: U matrix in csr format
	"""
	import scipy.sparse as sp

	GRID_CENTERS = grid_cell_centers(X_LIM, Y_LIM, GRID_SIZE)
	F = sp.kron(sp.identity(len(GRID_CENTERS)), numpy.array([1,1,1]))
	F = F.tocsc()
	v = numpy.array([1]*len(GRID_CENTERS))
	factor = cholmod.cholesky_AAt(F)
	FFT_inv_F = sp.lil_matrix((F.shape[0], F.shape[1]))
	for i in range(F.shape[1]):
		FFT_inv_F[:,i] = factor(F[:,i].toarray())
	Proj_matrix = F.T*FFT_inv_F
	Proj_perp_matrix = sp.identity(Proj_matrix.shape[0])-Proj_matrix
	v_bar = F.T*factor(v)

	return (Proj_perp_matrix, v_bar)


def sample_mult_normal_given_precision(mu, precision, num_of_samples):
	from scikits.sparse import cholmod
	factor = cholmod.cholesky(precision)
	D = factor.D(); D = numpy.reshape(D, (len(D),1))
	samples = factor.solve_Lt(numpy.random.normal(size=(len(mu),num_of_samples))/numpy.sqrt(D))
	samples = numpy.repeat(mu, num_of_samples, axis=1)+ factor.apply_Pt(samples)
	return samples

def conditional_sample(mu, precision, num_of_samples, proj_perp_matrix, v_bar):
	samples = sample_mult_normal_given_precision(0*mu, precision, num_of_samples)
	mu_shift = v_bar + proj_perp_matrix*mu
	for i in range(num_of_samples):
		temp = mu_shift + numpy.matrix(proj_perp_matrix*samples[:,i]).T
		samples[:,i] = numpy.array(temp).flatten()
	return samples

def mean_Gamma_all_players(Players, nu_sq_list, l_sq_list):
	K_inv = full_cov_matrix_inverse(nu_sq_list, l_sq_list)
	Gammas_list=[]
	num_grids_x, num_grids_y = grid_num_x_y(X_LIM, Y_LIM, GRID_SIZE)
	for player_id in Players.keys():
		Gamma_array = numpy.asarray([])
		Gammas = Players[player_id].Gammas
		for i in range(num_grids_x):
			for j in range(num_grids_y):
				Gamma_array = numpy.concatenate((Gamma_array, Gammas[(i,j)]), axis=0)
				
		Gammas_list.append(Gamma_array)

	Gammas_matrix = numpy.array(Gammas_list)
	mean_Gammas = numpy.mean(Gammas_matrix, axis=0)

	return mean_Gammas, K_inv

def update_Gamma_sampling(Players, sampling_parameters, PROJ_PERP_MATRIX, V_BAR, sigma_square):
	
	nu_sq_list = sampling_parameters['nu_square']
	l_sq_list = sampling_parameters['l_square']

	mean_Gammas, K_inv = mean_Gamma_all_players(Players, nu_sq_list, l_sq_list)
	
	for player_id in Players.keys():
		player = Players[player_id]
		W= player.make_sparse_diagnal_W()
		V = player.stack_V()
		
		posterior_mean, posterior_precision = get_posterior_params(W, numpy.matrix(V).T, numpy.matrix(mean_Gammas).T, K_inv, sigma_square)
		gamma_sample = conditional_sample(posterior_mean, posterior_precision, 
											1, PROJ_PERP_MATRIX, V_BAR)
		player.Gammas = update_player_Gamma(player, gamma_sample)

	return Players, mean_Gammas

if __name__ == "__main__":
	pass