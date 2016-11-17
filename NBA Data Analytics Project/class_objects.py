#!/usr/bin/env python
# encoding: utf-8
r"""
Module containing all class objects

:Author:
    Sheng Zhang -- Initial version (Aug 2016)
"""
import numpy
import scipy
import pdb
from utils import grid_num_x_y, parallel_update_sigma_square, rho_update
from energy_bonds import fista_algorithm
# ============================================================================
#  Possession Class: stores necessary information of one possession over
# 					 a game.
# ============================================================================

class Possession:
	def __init__(self, gamecode, poss_index, quarter, game_clock, offensive_team, defensive_team, offensive_players,\
				defensive_players, ball_location, offensive_locations, defensive_locations):
		
		self.gamecode = gamecode
		self.poss_index = poss_index
		self.quarter = quarter
		self.game_clock = game_clock
		self.offensive_team = offensive_team
		self.defensive_team = defensive_team
		self.offensive_players = offensive_players
		self.defensive_players = defensive_players
		self.ball_location = ball_location
		self.offensive_locations = offensive_locations
		self.defensive_locations = defensive_locations
		self.ball_states = None
		self.sampling_matchups_matrix = None        
		self.mean_location_matrix = None
		self.E_hidden_state_list = None
		self.E_two_hidden_states_product_list = None

# ============================================================================
#  Possession Player: stores necessary info of one player in the NBA league.
# 
# ============================================================================

class Player:
	def __init__(self, player_id, player_name, player_Gammas=None, grid_dependent=True):

		self.id = player_id
		self.name = player_name
		self.W = None
		self.V = None
		self.num_grid_datapoints=None

		if grid_dependent == True:       #if we are using grid dependent Gammas
			if player_Gammas==None:
				num_grids_x, num_grids_y = grid_num_x_y()
				self.Gammas = {}
				for i in range(num_grids_x):
					for j in range(num_grids_y):
						self.Gammas[(i,j)] = numpy.array([0.5,0.25,0.25])
			else:
				self.Gammas = player_Gammas
		
		else: #if we are using one Gammas over court 
			if player_Gammas == None:
				self.Gammas = numpy.array([0.5, 0.25, 0.25])
			else:
				self.Gammas = player_Gammas

	def make_sparse_diagnal_W(self):
		from scipy.sparse import block_diag

		num_grids_x, num_grids_y = grid_num_x_y()
		diag_sparse= []
		for i in range(num_grids_x):
			for j in range(num_grids_y):
				diag_sparse.append(self.W[(i,j)])

		diag_sparse = tuple(diag_sparse)

		diag_blocks_sparse = block_diag(diag_sparse, format='csc', dtype=numpy.float64)
		
		return diag_blocks_sparse
	
	def stack_V(self):
		
		num_grids_x, num_grids_y = grid_num_x_y()
		stack_vs = numpy.asarray([])
		for i in range(num_grids_x):
			for j in range(num_grids_y):
				stack_vs = numpy.concatenate((stack_vs, self.V[(i,j)]), axis=0)
	
		return stack_vs

# ============================================================================
#  Shared_Parameter Player: stores necessary info of one player in the NBA league.
# 
# ============================================================================
class Shared_Parameter:
	def __init__(self, sigma_square=None, energy_params=None, rho=None):
		
		if sigma_square is None:
			self.sigma_square = 5.0
		else:
			self.sigma_square = sigma_square
			
		if energy_params is None:
			self.energy_params = numpy.array([0.2, 0.3, 1.0, 1.5, 2.5])
		else:
			self.energy_params = energy_params

		if rho is None:
			self.rho = 0.8
		else:
			self.rho = rho

	def update_sigma_square_sampling(self, Possessions):

		total_distance_square, total_num_datapoints = parallel_update_sigma_square(Possessions)

		self.sigma_square = total_distance_square / total_num_datapoints

	def update_sigma_square_optimal(self, Possessions, min_fun_value):
		summation = 0.0
		for i in range(len(Possessions)):
			summation +=  sum([numpy.sum(Possessions[i].E_hidden_state_list[j]) for j in range(5)])

		self.sigma_square = 0.5 * min_fun_value / summation

	def update_energy_params(self, Possessions):

		self.energy_params = fista_algorithm(self.energy_params, Possessions, 0.01)

	def update_rho(self, Possessions):

		self.rho = rho_update(Possessions)


if __name__ == "__main__":
	pass