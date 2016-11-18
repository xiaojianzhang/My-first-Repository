#!/usr/bin/env python
# encoding: utf-8
r"""
Module containing functions for finding optimal Gammas

:Author:
    Sheng Zhang -- Initial version (Aug 2016)
"""
import numpy
import scipy
from utils import find_player_possessions, grid_location
X_LIM, Y_LIM = 47, 50
GRID_SIZE = 2
hoop_location = numpy.array([5.25, 25.0])

def build_array_for_optimization(offense, off_locations, def_locations, ball_location, E_hidden_state_list):

	X_x = numpy.zeros([1,3])
	X_y = numpy.zeros([1,3])
	num_seq = off_locations.shape[0]

	for n in range(num_seq):
		for def_num in range(5):
			X_x = numpy.concatenate((X_x, numpy.array([[off_locations[n,0], ball_location[n,0], hoop_location[0]]])), axis=0)
			X_y = numpy.concatenate((X_y, numpy.array([[off_locations[n,1], ball_location[n,1], hoop_location[1]]])), axis=0)
	
	X_x = X_x[1:,:] #shape=[5*num_seq,1]
	X_y = X_y[1:,:]	#shape=[5*num_seq,1]

	Y_x = numpy.asarray([])
	Y_y = numpy.asarray([])
	Omega_diag=[]
	for n in range(num_seq):
		for def_num in range(5):
			Omega_diag.append(numpy.sqrt(E_hidden_state_list[def_num][n,offense]))
			Y_x = numpy.concatenate((Y_x, numpy.array([def_locations[def_num][n,0]])), axis=0)
			Y_y = numpy.concatenate((Y_y, numpy.array([def_locations[def_num][n,1]])), axis=0)
			
	Omega = numpy.diag(tuple(Omega_diag))
	X_x = numpy.dot(Omega, X_x)
	X_y = numpy.dot(Omega, X_y)
	X = numpy.concatenate(X_x, X_y, axis=0)


	Y_x = numpy.dot(Omega, Y_x)
	Y_y = numpy.dot(Omega, Y_y)
	Y = numpy.concatenate(Y_x, Y_y, axis=0)

	return X, Y

def find_X_Y_in_constrained_LS(Possession, player_id):

	offensive_players = Possession.offensive_players
	offense = offensive_players.index(player_id)
	offense_locations = Possession.offensive_locations[offense,:,:] #shape=[num_seq,2]
	defense_locations = Possession.defensive_locations #shape=[5,num_seq,2]
	ball_location = Possession.ball_location
	
	X,Y = build_array_for_optimization(offense, offense_locations, defense_locations, ball_location, Possession.E_hidden_state_list)
	
	return X, Y

def find_optimal_Gamma_fun_value(X, Y):

		X_T_X = numpy.dot(X.T, X)
		X_T_Y = numpy.dot(X.T, Y)
		lhs_matrix = numpy.concatenate((2*X_T_X, numpy.array([[1.0],[1.0],[1.0]])), axis=1)
		lhs_matrix = numpy.concatenate((lhs_matrix, numpy.array([[1.0,1.0,1.0,0.0]])), axis=0)
		rhs_array = numpy.concatenate((2*X_T_Y, numpy.ones(1)), axis=0)
		solution = numpy.linalg.solve(lhs_matrix, rhs_array)
		Gamma_new = solution[:3]

		if ((Gamma_new > 0.0).all() and (Gamma_new < 1.0).all()) == False:
			from cvxopt import matrix, solvers
			Q = 2*matrix(X_T_X)
			P = matrix(-2*X_T_Y)
			G = matrix([[-1.0,0.0,0.0],[0.0,-1.0,0.0],[0.0, 0.0, -1.0]])
			h = matrix([0.0,0.0, 0.0])
			A = matrix([1.0, 1.0, 1.0], (1,3))
			b = matrix(1.0)
			sol=solvers.qp(Q, P, G, h, A, b)
			Gamma_new = numpy.array([sol['x'][i] for i in range(3)])

		min_fun_value = numpy.sum((Y - numpy.dot(X, Gamma_new))**2)

		return Gamma_new, min_fun_value

def player_dependent_Gammas_and_sigma_square_update(Possessions, Players):
	min_fun_value = 0
	player_possessions_dict = find_player_possessions(Possessions)
	for player_id in Players.keys():
		poss_index_list = player_possessions_dict[player_id]
		X = numpy.zeros([1,3])
		Y = numpy.asarray([])
		for k in poss_index_list:
			X_poss,Y_poss= find_X_Y_in_constrained_LS(Possessions[k], player_id)
			X = numpy.concatenate(X, X_poss, axis=0)
			Y = numpy.concatenate(Y, Y_poss, axis=0)
		X = X[1:,:]
		Gammas_new, optimal_fun_value = find_optimal_Gamma_fun_value(X, Y)
		Players[player_id].Gammas = Gammas_new
		min_fun_value += optimal_fun_value

	return Players, min_fun_value

def find_X_Y_in_constrained_LS_grid_dependent(Possessions, grid_datapoint_loc):

	num_grids_x, num_grids_y = grid_num_x_y(X_LIM, Y_LIM, GRID_SIZE)
	grid_X = {}
	grid_Y = {}
	for i in range(num_grids_x):
		for j in range(num_grids_y):
			grid_datapoint = grid_datapoint_loc[(i,j)]
			if len(grid_datapoint) != 0:
				X, Y = build_array_for_optimization_grid_dependent(Possessions, grid_datapoint)
				grid_X[(i,j)] = X
				grid_Y[(i,j)] = Y
			else:
				grid_X[(i,j)] = None
				grid_Y[(i,j)] = None

	return grid_X, grid_Y

def build_array_for_optimization_grid_dependent(Possessions, grid_datapoint):
	X_x = numpy.zeros([1,3])
	X_y = numpy.zeros([1,3])
	for k, n, offense in grid_datapoint:
		off_locations = Possessions[k].offensive_locations[offense,:,:]
		ball_location = Possessions[k].ball_location
		for def_num in range(5):
			X_x = numpy.concatenate((X_x, numpy.array([[off_locations[n,0], ball_location[n,0], hoop_location[0]]])), axis=0)
			X_y = numpy.concatenate((X_y, numpy.array([[off_locations[n,1], ball_location[n,1], hoop_location[1]]])), axis=0)
	
	X_x = X_x[1:,:] #shape=[5*len(grid_datapoint),1]
	X_y = X_y[1:,:]	#shape=[5*len(grid_datapoint),1]

	Y_x = numpy.asarray([])
	Y_y = numpy.asarray([])
	Omega_diag=[]
	for k, n, offense in grid_datapoint:
		E_hidden_state_list = Possessions[k].E_hidden_state_list
		def_locations = Possessions[k].defensive_locations
		for def_num in range(5):
			Omega_diag.append(numpy.sqrt(E_hidden_state_list[def_num][n,offense]))
			Y_x = numpy.concatenate((Y_x, numpy.array([def_locations[def_num][n,0]])), axis=0)
			Y_y = numpy.concatenate((Y_y, numpy.array([def_locations[def_num][n,1]])), axis=0)
			
	Omega = numpy.diag(tuple(Omega_diag))
	X_x = numpy.dot(Omega, X_x)
	X_y = numpy.dot(Omega, X_y)
	X = numpy.concatenate(X_x, X_y, axis=0)


	Y_x = numpy.dot(Omega, Y_x)
	Y_y = numpy.dot(Omega, Y_y)
	Y = numpy.concatenate(Y_x, Y_y, axis=0)

	return X, Y

def find_all_offense_loc_given_grid(Possessions):
	'''
		given a grid index (i,j), find all offense location on that grid
	'''
	grid_datapoint_loc={}
	num_grids_x, num_grids_y = grid_num_x_y(X_LIM, Y_LIM, GRID_SIZE)
	for i in range(num_grids_x):
		for j in range(num_grids_y):
			grid_datapoint_loc[(i,j)]=[]

	for k in range(len(Possessions)):
		offense_locations = Possessions[k].offensive_locations
		grid_location_list = [grid_location(offense_locations[i,:,:]) for i in range(5)]		

		for offense in range(5):
			for n in range(offense_locations.shape[1]):
				grid_loc = grid_location_list[offense][n]
				grid_datapoint_loc[grid_loc].append((k, n, offense)) #(Possession[k], seq, offense)

	return grid_datapoint_loc

def player_independent_Gammas_and_sigma_square_update_grid_dependent(Possessions, player):
	min_fun_value = 0
	num_grids_x, num_grids_y = grid_num_x_y(X_LIM, Y_LIM, GRID_SIZE)
	grid_datapoint_loc = find_all_offense_loc_given_grid(Possessions)

	grid_X, grid_Y = find_X_Y_in_constrained_LS_grid_dependent(Possessions, grid_datapoint_loc)
	
	for i in range(num_grids_x):
		for j in range(num_grids_y):
			if grid_Y[(i,j)] not None: 
				X = grid_X[(i,j)]
				Y = grid_Y[(i,j)]
				Gammas_new, optimal_fun_value = find_optimal_Gamma_fun_value(X, Y)
				player.Gammas[(i,j)] = Gammas_new
				min_fun_value += optimal_fun_value

	return player, min_fun_value


if __name__ == "__main__":
	pass