'''
	The script is used for sampling from the posterior distribution for HMM hidden states.

	reference: 'Sampling from the posterior distribution for HMM hidden states.' by Christophe Rhodes. 2008.
'''

'''
Code for the section "Who is guarding who", by Sheng Zhang, Sun July 24, 2016.

This code is based on paper "" by 

and book "Pattern Recognition and Machine Learning" by Bishop.
'''
import numpy
import os
import csv
import lxml.etree as ET
from collections import namedtuple
import random
import scipy
import timeit
import pickle
import multiprocessing
from scikits.sparse import cholmod
import time 

season_year = "2015-2016"
X_LIM, Y_LIM = 50, 47
GRID_SIZE = 2
hoop_location = numpy.array([5.25, 25.0])

class Shared_Parameter:
	def __init__(self, sigma_square=None, rho=None):
		
		if sigma_square is None:
			self.sigma_square = 5.0
		else:
			self.sigma_square = sigma_square
			
		if rho is None:
			self.rho = 0.8
		else:
			self.rho = rho
			
	def transition_matrix(self):
		
		trans_matrix = numpy.ones([5,5]) * ((1-self.rho)/4.0)
		numpy.fill_diagonal(trans_matrix, self.rho)
		
		return trans_matrix
	
	def update_rho(self, possession_list):

		summation1 = 0.0
		summation2 = 0.0
		for i in range(len(possession_list)):
			for j in range(5):
				for k in range(5):
					summation1 = summation1 + sum([sum(possession_list[i].E_two_hidden_states_product_list[j][:,k,kk]) for kk in range(5) if kk == k])
					summation2 = summation2 + sum([sum(possession_list[i].E_two_hidden_states_product_list[j][:,k,kk]) for kk in range(5) if kk != k])


		#update parameter rho
		Q = 0.25 * summation1/summation2
		rho_update = Q/(1+Q)

		self.rho = rho_update

	def update_sigma_square(self, Possessions):

		total_variance = 0
		total_num_datapoints = 0
		for possession in Possessions:
			variance, num_datapoints = compute_variance(possession)
			total_variance += variance
			total_num_datapoints += num_datapoints

		self.sigma_square = total_variance / total_num_datapoints

def compute_variance(Possession):

	mean_location_list = Possession.mean_location_list
	defensive_locations = Possession.defensive_locations
	sampling_hidden_states_list = Possession.sampling_hidden_states_list
	N = len(mean_location_list[0])
	variance = 0
	num_datapoints = N * 5
	for seq in range(N):
		for def_num in range(5):
			matchup = sampling_hidden_states_list[def_num][seq]
			def_loc = defensive_locations[def_num, seq, :]
			mean_location = mean_location_list[matchup][seq]
			variance += numpy.sum((def_loc - mean_location)**2)

	return variance, num_datapoints

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
		self.init_hidden_state_dist = init_hidden_state_dist()
		self.on_ball_of_ball_matrix=None
		self.sampling_hidden_states_list=None        
		self.likelihood_matrix_list=None
		self.mean_location_list=None
		self.E_two_hidden_states_product_list=None
		self.log_likehood_list = None

class Player:
	def __init__(self, player_id, player_name, player_Gammas=None):

		self.id = player_id
		self.name = player_name
		self.W_onball = None
		self.V_onball = None
		self.W_offball = None
		self.V_offball = None
		self.num_seq_grid_W_onball = None
		self.num_seq_grid_V_onball = None
		self.num_seq_grid_W_offball = None
		self.num_seq_grid_V_offball = None
		self.sparse_matrix_W_onball=None
		self.sparse_matrix_W_offball=None
		self.stackarays_V_onball=None
		self.stackarays_V_offball=None
		if player_Gammas==None:
			num_grids_x, num_grids_y = grid_num_x_y()
			self.Gammas = {}
			for i in range(num_grids_x):
				for j in range(num_grids_y):
					self.Gammas[(i,j)] = {"on_ball":numpy.array([0.5,0.25,0.25]),\
										  "off_ball":numpy.array([0.5,0.25,0.25])}

	def make_sparse_diagnal_W(self):
		
		num_grids_x, num_grids_y = grid_num_x_y()
		diag_sparse_onball = [[]]
		diag_sparse_offball = [[]]
		for i in range(num_grids_x):
			for j in range(num_grids_y):
				diag_sparse_onball = scipy.linalg.block_diag(diag_sparse_onball, self.W_onball[(i,j)])
				diag_sparse_offball = scipy.linalg.block_diag(diag_sparse_offball, self.W_offball[(i,j)])
		
		return diag_sparse_onball, diag_sparse_offball
	
	def stack_V(self):
		
		num_grids_x, num_grids_y = grid_num_x_y()
		stack_vs_onball = numpy.asarray([])
		stack_vs_offball = numpy.asarray([])
		for i in range(num_grids_x):
			for j in range(num_grids_y):
				stack_vs_onball = numpy.concatenate((stack_vs_onball, self.V_onball[(i,j)]),axis=0)
				stack_vs_offball = numpy.concatenate((stack_vs_offball, self.V_offball[(i,j)]),axis=0)
	
		return stack_vs_onball, stack_vs_offball


def init_hidden_state_dist(E_hidden_state_list=None):
	if E_hidden_state_list == None:
		return numpy.ones([5,5]) * 0.2
	
	else:  #update init_hidden_state_dist
		init_hidden_state_dist = numpy.zeros([5,5])
		
		for i,E_hidden_state in enumerate(E_hidden_state_list):
			init_hidden_state_dist[i,:] = E_hidden_state[0,:] / numpy.sum(E_hidden_state[0,:])
			
		return init_hidden_state_dist


def data_format_transform(data, name):
	
	if name == "ball":
		location = data.strip("( )").split(",")
		if location[0].strip(" ")[0] == "-":
			x = -float(location[0].strip(" ")[1:])
		else:
			x = float(location[0].strip(" "))
		
		if location[1].strip(" ")[0] == "-":
			y = -float(location[1].strip(" ")[1:])
		else:
			y = float(location[1].strip(" "))
			
		z = float(location[2].strip(" "))
		return x,y,z
	
	if name == "player":
		info = data.strip("( )").split(",")
		id = int(info[0].strip(" "))
		if info[1].strip(" ")[0] == "-":
			x = -float(info[1].strip(" ")[1:])
		else:
			x = float(info[1].strip(" "))
			
		if info[2].strip(" ")[0] == "-":
			y = -float(info[2].strip(" ")[1:])
		else:
			y = float(info[2].strip(" "))
			
		return id,x,y


def get_data_poss_based(gamecode, poss_index, season_year):
	
	for game_file in os.listdir("./possessions_sequence_optical/"+season_year+"/"):
		if game_file[:10] == gamecode:
			sequence_optical_poss_bassed_path = "./possessions_sequence_optical/"+season_year+"/"+game_file
	
	f = open(sequence_optical_poss_bassed_path, "r", newline=None)
	csv_f = csv.reader(f, quotechar='"', delimiter = ',')
	
	rows=[]
	for row in csv_f:
		if row[3] == poss_index:
			rows.append(row)
			
	num_seq = len(rows)
	quarter = rows[-1][0]
	game_clock = []
	offensive_team = rows[0][5]
	defensive_team = rows[0][11]
	offensive_players = []
	defensive_players = []
	offensive_locations = numpy.zeros([5, num_seq, 2])
	defensive_locations = numpy.zeros([5, num_seq, 2])
	ball_location = numpy.zeros([num_seq, 2])
	
	for i in range(num_seq):
		game_clock.append(rows[i][1])
		ball_data = rows[i][4]
		x,y,z = data_format_transform(ball_data, "ball")
		ball_location[i,:] = numpy.array([x,y])
		for j in range(5):
			off_data = rows[i][6+j]
			off_id, off_x, off_y = data_format_transform(off_data, "player")
			offensive_locations[j, i, :] = numpy.array([off_x, off_y])
			if str(off_id) not in offensive_players:
				offensive_players.append(str(off_id))
			def_data = rows[i][12+j]
			def_id, def_x, def_y = data_format_transform(def_data, "player")
			defensive_locations[j, i, :] = numpy.array([def_x, def_y])
			if str(def_id) not in defensive_players:
				defensive_players.append(str(def_id))
			
	return quarter, game_clock, offensive_team, defensive_team, offensive_players, defensive_players,\
		   ball_location, offensive_locations, defensive_locations

def likelihood(mean_location, y, **dist_params):
	#default distribution is Gaussian with mean= mean_location, variance = sigma_square
	from scipy.stats import multivariate_normal

	llh = multivariate_normal.pdf(y, mean=mean_location, cov=dist_params["sigma_square"])

	return llh


def mean_location_player_poss(player_location, ball_location, player_Gamma, on_ball_off_ball):
	num_seq = player_location.shape[0]
	mean_location = numpy.zeros([num_seq, 2])

	for i in range(num_seq): #for each sequence in that possession
		grid_loc = grid_location(player_location[i])  #grid_location e.g., (4,5)
		if on_ball_off_ball[i] == 1:
			on_or_off = "on_ball"
		else:
			on_or_off = "off_ball"
		Gamma = player_Gamma[grid_loc][on_or_off]     #player_Gamma[(4,5)]["on_ball"]
		X = numpy.stack((player_location[i], ball_location[i], hoop_location), axis=1) #X=[2 by 3 matrix]
		mean_location[i,:] = numpy.dot(X, Gamma)

	return mean_location #array, shape=(num_seq, 2)


def likelihood_matrix_poss(mean_location_list, def_location, **dist_params):
	num_seq = def_location.shape[0]
	likelihood_matrix = numpy.zeros([num_seq, 5])

	for i in range(5):
		for j in range(num_seq):
			likelihood_matrix[j,i]=likelihood(mean_location_list[i][j], def_location[j],\
										   sigma_square=dist_params["sigma_square"])

	return likelihood_matrix

def sampling_from_posterior_dist(def_num, likelihood_matrix, alpha, transition_matrix):

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

def indicator(game_clock, poss_period, on_air_period, off_players):
	for item in poss_period:
		if game_clock <= item[0] and game_clock >= item[1]:
			index = off_players.index(item[2])
			indicator = numpy.zeros((5,),dtype=numpy.int)
			indicator[index] = 1
			return indicator
	
	if len(on_air_period) != 0:    
		for item in on_air_period:
			if game_clock < item[0] and game_clock > item[1]:
				indicator = numpy.zeros((5,),dtype=numpy.int)
				return indicator

def find_pass_possession_moment_player(possession):
	pass_moment_player = []
	poss_moment_player = []
	quarter = possession.quarter
	game_clock = possession.game_clock
	gamecode = possession.gamecode
	poss_index = possession.poss_index
	file_path = "./NBA_EVENNBA_FINAL_SEQUENCE_PBP_OPTICAL/NBA_FINAL_SEQUENCE_PBP_OPTICAL$"\
				+ gamecode + ".XML"
	root = ET.parse(file_path).getroot()
	
	shot_time_game_clock = game_clock[-1]
	if "." not in shot_time_game_clock: #in case game-clock = integer
		shot_time_game_clock = shot_time_game_clock + ".00"
	if len(shot_time_game_clock.split(".")[1]) < 2: #in case game-clock is not two decimal precision
		shot_time_game_clock = shot_time_game_clock + "0"
		
	start_of_possession = game_clock[0]
	
	ele = root.find(".//sequence-pbp[@period='{}']/moment[@game-clock='{}']".format\
			  (quarter, shot_time_game_clock)) #starting from the shot time
	player_id = ele.attrib['global-player-id']
	shooter_id = ele.attrib['global-player-id']
	event_id = ele.attrib['event-id']
	
	ele = ele.getprevious() #previous time step
	previous_game_clock = float(ele.attrib['game-clock'])
	game_clock_now = float(shot_time_game_clock)
	while(previous_game_clock <= float(start_of_possession) and\
		  previous_game_clock > float(game_clock_now)):
		
		player_id = ele.attrib['global-player-id']
		event_id = ele.attrib['event-id']
		game_clock_now = previous_game_clock
			
		if event_id == '23': #'23' represents 'Possession'
			poss_moment_player.insert(0, (game_clock_now, player_id))

		elif event_id == '22' or event_id == '25':#'22' represents 'Pass', '25' represents 'Assist'
			pass_moment_player.insert(0, (game_clock_now, player_id))
			
		ele = ele.getprevious()
		if ele is None:
			break
		else:
			previous_game_clock = float(ele.attrib['game-clock'])
			
	if ele is not None and float(ele.attrib['game-clock']) > float(game_clock_now):
		player_id = ele.attrib['global-player-id']
		event_id = ele.attrib['event-id']
		game_clock = ele.attrib['game-clock']
		if event_id == '23':
			poss_moment_player.insert(0, (game_clock, player_id))
			
		elif event_id == '22' or event_id == '25':
			pass_moment_player.insert(0, (game_clock, player_id))
			
	if len(poss_moment_player) == 0: #if no pass happens within the possession
		poss_moment_player.append((start_of_possession, shooter_id))
	
	return pass_moment_player, poss_moment_player

def identify_ball_status(possession):
	poss_period = []
	on_air_period = []
	pass_moment_player, poss_moment_player = find_pass_possession_moment_player(possession)
	game_clock = possession.game_clock
	
	if len(pass_moment_player) == 0 and len(poss_moment_player) == 1:
		if float(game_clock[0]) > float(poss_moment_player[0][0]):
			on_air_period.append((float(game_clock[0]), float(poss_moment_player[0][0])))
			poss_period.append((float(poss_moment_player[0][0]), float(game_clock[-1]), poss_moment_player[0][1]))
		
		else:
			poss_period.append((float(game_clock[0]), float(game_clock[-1]), poss_moment_player[0][1]))
	
	else:
		
		if float(pass_moment_player[0][0]) > float(poss_moment_player[0][0]):
			poss_period.append((float(game_clock[0]), float(pass_moment_player[0][0]), pass_moment_player[0][1]))

			for i in range(len(pass_moment_player)):
				on_air_period.append((float(pass_moment_player[i][0]), float(poss_moment_player[i][0])))

			for i in range(len(pass_moment_player)-1):
				poss_period.append((float(poss_moment_player[i][0]), float(pass_moment_player[i+1][0]),\
									poss_moment_player[i][1]))

			poss_period.append((float(poss_moment_player[-1][0]), float(game_clock[-1]), poss_moment_player[-1][1]))
		
		elif float(pass_moment_player[0][0]) < float(poss_moment_player[0][0]):
			on_air_period.append((float(game_clock[0]), float(poss_moment_player[0][0])))
			
			for i in range(len(poss_moment_player)-1):
				poss_period.append((float(poss_moment_player[i][0]), float(pass_moment_player[i][0]),\
									poss_moment_player[i][1]))
				
			for i in range(len(poss_moment_player)-1):
				on_air_period.append((float(pass_moment_player[i][0]), float(poss_moment_player[i+1][0])))

			poss_period.append((float(poss_moment_player[-1][0]), float(game_clock[-1]), poss_moment_player[-1][1]))
	
	return poss_period, on_air_period

def indicator(game_clock, poss_period, on_air_period, off_players):
	for item in poss_period:
		if game_clock <= item[0] and game_clock >= item[1]:
			index = off_players.index(item[2])
			indicator = numpy.zeros((5,),dtype=numpy.int)
			indicator[index] = 1
			return indicator
	
	if len(on_air_period) != 0:    
		for item in on_air_period:
			if game_clock <= item[0] and game_clock >= item[1]:
				indicator = numpy.zeros((5,),dtype=numpy.int)
				return indicator
			
def on_ball_off_ball_air(possession):
	poss_period, on_air_period = identify_ball_status(possession)
	game_clock = possession.game_clock
	num_seq = len(game_clock)
	off_players = possession.offensive_players
	on_ball_off_ball = numpy.zeros((num_seq, 5),dtype=numpy.int)
	for i in range(num_seq):
		on_ball_off_ball[i,:] = indicator(float(game_clock[i]), poss_period, on_air_period, off_players)
	
	return on_ball_off_ball

def forward_backward_rescaled_sampling(def_num, likelihood_matrix, initial_hidden_state_dist, transition_matrix):

	num_seq = likelihood_matrix.shape[0]
	#initialization:
	alpha = numpy.zeros([num_seq,5])
	alpha_rescaled = numpy.zeros([num_seq,5])
	beta_rescaled = numpy.zeros([num_seq,5])
	c = []
	for k in range(5):
		alpha[0,k] = initial_hidden_state_dist[def_num, k] * likelihood_matrix[0,k]
	c.append(sum(alpha[0,:]))
	alpha_rescaled[0,:] = alpha[0,:] / c[0]
	beta_rescaled[-1,:] = numpy.ones(5)

	#iterations:
	for n in range(1,num_seq):
		for k in range(5):
			alpha[n,k] = likelihood_matrix[n,k] * numpy.dot(alpha_rescaled[n-1,:], transition_matrix[k,:])
		c.append(sum(alpha[n,:]))
		alpha_rescaled[n,:] = alpha[n,:] / c[n]

	for n in range(num_seq-2,-1,-1):
		for k in range(5):
			beta_rescaled[n,k] = sum(beta_rescaled[n+1,:] * likelihood_matrix[n+1,:] * transition_matrix[k,:]) / c[n+1]
	
	#Evaluations:
	log_likelihood = sum(numpy.log(c))  #log(P(D_j1, D_j2,...,D_jN| parameters))

	#E_hidden_state = numpy.empty([num_seq,5])
	E_two_hidden_states_product = numpy.empty([num_seq-1,5,5])

	#E_hidden_state = alpha_rescaled * beta_rescaled

	for n in range(1,num_seq,1):
		for k in range(5):#index for I_n
			for kk in range(5):#index for I_(n-1)
				E_two_hidden_states_product[n-1, k, kk] = c[n] * alpha_rescaled[n-1,kk] * likelihood_matrix[n,k] * transition_matrix[k,kk] * beta_rescaled[n,k]

	sample_list = sampling_from_posterior_dist(def_num, likelihood_matrix, alpha, transition_matrix)

	return E_two_hidden_states_product, log_likelihood, sample_list

def parallel_forward_backward(possession):

	E_two_hidden_states_product_list=[]
	log_likelihood_list=[]
	sampling_hidden_states_list=[]

	pool = multiprocessing.Pool(processes=5)
	results = [pool.apply_async(forward_backward_rescaled_sampling,\
			  args=(def_num, possession.likelihood_matrix_list[def_num], possession.init_hidden_state_dist, global_parameter.transition_matrix())) for def_num in range(5)]

	pool.close()
	pool.join()
	output = [p.get() for p in results]
	output.sort()
	for j in range(5):
		E_two_hidden_states_product_list.append(output[j][1])
		log_likelihood_list.append(output[j][2])
		sampling_hidden_states_list.append(output[j][3])

	return E_two_hidden_states_product_list, log_likelihood_list, sampling_hidden_states_list 


def E_step(team, Possessions, Players, global_parameter):

	for k in range(len(Possessions)):
		if k % 200 == 0:
			print("{} possessions finished".format(k))

		likelihood_matrix_list=[]
		mean_location_list=[]
		off_location = Possessions[k].offensive_locations
		ball_location = Possessions[k].ball_location
		offensive_players = Possessions[k].offensive_players
		on_ball_off_ball = Possessions[k].on_ball_of_ball_matrix
		
		for i in range(5):
			player = Players[offensive_players[i]]
			mean_location_list.append(mean_location_player_poss(off_location[i,:,:], ball_location,\
																player.Gammas, on_ball_off_ball[:,i]))
		for j in range(5):
			def_location = Possessions[k].defensive_locations[j,:,:]
			likelihood_matrix_list.append(likelihood_matrix_poss(mean_location_list, def_location,\
																 sigma_square=global_parameter.sigma_square))
		Possessions[k].mean_location_list = mean_location_list
		Possessions[k].likelihood_matrix_list = likelihood_matrix_list
		
		sampling_hidden_states_list=[]
		E_two_hidden_states_product_list = []
		log_likelihood_list = []
		for def_num in range(5):
			E_two_hidden_states_product, log_likelihood, sampling_hidden_states \
				= forward_backward_rescaled_sampling(def_num, likelihood_matrix_list[def_num], Possessions[k].init_hidden_state_dist, global_parameter.transition_matrix())
			sampling_hidden_states_list.append(sampling_hidden_states)
			E_two_hidden_states_product_list.append(E_two_hidden_states_product)
			log_likelihood_list.append(sampling_hidden_states)
		
		Possessions[k].sampling_hidden_states_list = sampling_hidden_states_list
		Possessions[k].E_two_hidden_states_product_list = E_two_hidden_states_product_list
		Possessions[k].log_likelihood_list = log_likelihood_list
				
	total_log_likelihood = numpy.sum([numpy.sum(poss.log_likelihood_list) for poss in Possessions])

	print("process for {} has been finished".format(team))

	return team, Possessions, total_log_likelihood

def parallel_E_step(pickle_data_name_list, Players, global_parameter):
	
	num_processes = len(pickle_data_name_list)
	Possessions_list = []

	for data_name in pickle_data_name_list:
		team = data_name.split("_")[0]
		f = open("./input_data/"+data_name, 'rb')
		Possessions_list.append((team, pickle.load(f)))

	pool = multiprocessing.Pool(processes=num_processes)
	results = [pool.apply_async(E_step,\
			  args=(team, Possessions, Players, global_parameter)) for team, Possessions in Possessions_list]

	pool.close()
	pool.join()
	output = [p.get() for p in results]
	total_log_likelihood_new = 0
	Possessions_new = []
	for i,result in enumerate(output):
		team = result[0]
		Possessions_new += result[1]
		name = "{}_Possessions_new.pickle".format(team)
		file = open("./intermediate_data/"+name, 'wb')
		pickle.dump(result[1], file)
		total_log_likelihood_new += result[2]

	return Possessions_new, total_log_likelihood_new

def parallel_E_step_2(Possessions, Players, global_parameter):
	
	Possessions_list = []
	length = len(Possessions)
	quotient, remainder = divmod(length, 24)
	for i in range(0, 23*quotient, quotient):
		Possessions_list.append(Possessions[i:i+quotient])
		
	Possessions_list.append(Possessions[23*quotient:])

	pool = multiprocessing.Pool(processes=24)
	results = [pool.apply_async(E_step,\
			  args=(Possessions, Players, global_parameter)) for Possessions in Possessions_list]

	pool.close()
	pool.join()
	output = [p.get() for p in results]
	total_log_likelihood_new = 0
	Possessions_new = []
	for i,result in enumerate(output):
		Possessions_new += result[0]
		total_log_likelihood_new += result[1]

	return Possessions_new, total_log_likelihood_new

def grid_location(player_location):

	if player_location.ndim == 1:
		
		#player_location, array, shape=(2, )
		loc_x = player_location[0] // GRID_SIZE
		loc_y = player_location[1] // GRID_SIZE
		
		if player_location[0] < 0:
			loc_x = 0
		if player_location[1] < 0:
			loc_y = 0
		if player_location[0] >= X_LIM:
			loc_x = (X_LIM // GRID_SIZE) - 1    
		if player_location[1] >= Y_LIM:
			loc_y = (Y_LIM // GRID_SIZE) - 1 
		
		return (int(loc_x), int(loc_y))

	else:
	#player_location, array, shape=(N, 2)
		grid_loc_list = []
		for player_loc in player_location:
			grid_loc_list.append(grid_location(player_loc))
	
	return grid_loc_list

def grid_num_x_y():
	
	num_grids_x = int(numpy.ceil(X_LIM / GRID_SIZE))
	num_grids_y = int(numpy.ceil(Y_LIM / GRID_SIZE))
	
	return num_grids_x, num_grids_y

def grid_cell_centers():

	num_grids_x, num_grids_y = grid_num_x_y()
	delta = GRID_SIZE / 2.0
	
	center_x = numpy.arange(delta, X_LIM, GRID_SIZE)
	if (num_grids_x * GRID_SIZE) > X_LIM:
		last_second = (num_grids_x - 1) * GRID_SIZE
		center_x = list(center_x)
		center_x.append(last_second + (X_LIM - last_second) / 2.0)
		
	center_y = numpy.arange(delta, Y_LIM, GRID_SIZE)
	if (num_grids_y * GRID_SIZE) > Y_LIM:
		last_second = (num_grids_y - 1) * GRID_SIZE
		center_y = list(numpy.arange(delta, Y_LIM, GRID_SIZE))
		center_y.append(last_second + (Y_LIM - last_second) / 2.0)
		
	centers_list = []
	for x in center_x:
		for y in center_y:
			centers_list.append((x,y)) 
	
	return centers_list

def Calculate_W_V(Possessions, Players, player_possessions_dict):
	for player_id in Players.keys():
		poss_index_list = player_possessions_dict[player_id]
		num_grids_x, num_grids_y = grid_num_x_y()
		W_onball = {}
		W_offball = {}
		num_seq_grid_onball_W={}
		num_seq_grid_offball_W={}
		V_onball = {}
		V_offball= {}
		num_seq_grid_onball_V = {}
		num_seq_grid_offball_V = {}
		for i in range(num_grids_x):
			for j in range(num_grids_y):
				W_onball[(i,j)] = numpy.zeros((3,3))
				num_seq_grid_onball_W[(i,j)] = 0
				W_offball[(i,j)] = numpy.zeros((3,3))
				num_seq_grid_offball_W[(i,j)] = 0
				V_onball[(i,j)] = numpy.zeros(3)
				num_seq_grid_onball_V[(i,j)] = 0
				V_offball[(i,j)] = numpy.zeros(3)
				num_seq_grid_offball_V[(i,j)] = 0
		for k in poss_index_list:
			possession = Possessions[k]
			index = possession.offensive_players.index(player_id)
			off_location = possession.offensive_locations[index]
			ball_location = possession.ball_location
			matchups_list = possession.sampling_hidden_states_list
			def_location = possession.defensive_locations
			on_ball_of_ball_matrix = possession.on_ball_of_ball_matrix
			for i in range(off_location.shape[0]):
				grid_loc = grid_location(off_location[i])
				Z = numpy.vstack((off_location[i], ball_location[i], hoop_location))

				if on_ball_of_ball_matrix[i,index] == 1:#on ball
					W_onball[grid_loc] += numpy.dot(Z, Z.T)
					num_seq_grid_onball_W[grid_loc] += 1
					for j in range(5):
						if matchups_list[j][i] == index:
							d = def_location[j, i, :]
							num_seq_grid_onball_V[grid_loc] += 1
							V_onball[grid_loc] += numpy.dot(Z, d)
				else:#off ball
					W_offball[grid_loc] += numpy.dot(Z, Z.T)
					num_seq_grid_offball_W[grid_loc] += 1
					for j in range(5):
						if matchups_list[j][i] == index:
							d = def_location[j, i, :]
							num_seq_grid_offball_V[grid_loc] += 1
							V_offball[grid_loc] += numpy.dot(Z, d)
					
		for grid in V_onball.keys():
			if num_seq_grid_onball_W[grid] != 0:
				W_onball[grid] /= float(num_seq_grid_onball_W[grid])
			
			if num_seq_grid_offball_W[grid] != 0:
				W_offball[grid] /= float(num_seq_grid_offball_W[grid])

			if num_seq_grid_onball_V[grid] != 0:
				V_onball[grid] /= float(num_seq_grid_onball_V[grid])

			if num_seq_grid_offball_V[grid] != 0:
				V_offball[grid] /= float(num_seq_grid_offball_V[grid])

		Players[player_id].W_onball = W_onball
		Players[player_id].num_seq_grid_W_onball = num_seq_grid_onball_W
		Players[player_id].W_offball = W_offball
		Players[player_id].num_seq_grid_W_offball = num_seq_grid_offball_W
		
		Players[player_id].V_onball = V_onball
		Players[player_id].num_seq_grid_V_onball = num_seq_grid_onball_V
		Players[player_id].V_offball = V_offball
		Players[player_id].num_seq_grid_V_offball = num_seq_grid_offball_V
		
	return Players

def find_player_possessions(Possessions):

	player_poss_dict = {}
	for i,poss in enumerate(Possessions):
		offensive_players = Possessions[i].offensive_players
		for player_id in offensive_players:
			if player_id not in player_poss_dict.keys():
				player_poss_dict[player_id]=[i]
			else:
				player_poss_dict[player_id].append(i)

	return player_poss_dict

def collecting_possession_data_by_team(team, poss_gameid_num_list):
	possession_list=[]#Possession class list
	for gamecode, poss_num in poss_gameid_num_list:
		quarter, game_clock, offensive_team, defensive_team, offensive_players, defensive_players,\
		ball_location, offensive_locations, defensive_locations = get_data_poss_based(gamecode, str(poss_num), season_year)
		possession = Possession(gamecode, str(poss_num), quarter, game_clock, offensive_team, defensive_team, offensive_players,\
		defensive_players, ball_location, offensive_locations, defensive_locations)

		try:
			possession.on_ball_of_ball_matrix = on_ball_off_ball_air(possession)
		except:
			print((gamecode, poss_num))
			pass
		else:
			possession_list.append(possession)
		
	return team, possession_list

def parallely_collecting_possession_data(common_game_poss_dict):
	Possessions = []
	team_list = list(common_game_poss_dict.keys())

	pool = multiprocessing.Pool(processes=30)
	results = [pool.apply_async(collecting_possession_data_by_team,\
			  args=(team, common_game_poss_dict[team][2])) for team in team_list]

	pool.close()
	pool.join()
	output = [p.get() for p in results]
	for result in output:
		team = result[0]
		Possessions = result[1]
		file = open("./input_data/{}_Possessions.pickle".format(team),'wb')
		pickle.dump(Possessions, file)


def update_player_Gamma(Player, Gammas_sample, on_or_off):
	Gammas_dict = Player.Gammas
	GRID_CENTERS = grid_cell_centers()
	N = len(GRID_CENTERS)
	num_grids_x, num_grids_y = grid_num_x_y()
	Gammas = Gammas_sample.reshape((N,3))
	row = 0
	for i in range(num_grids_x):
		for j in range(num_grids_y):
			Gammas_dict[(i,j)][on_or_off] = Gammas[row,:]
			row += 1

	return Gammas_dict

def covariance_calculation(samples_matrix):

	covariance_list = []
	covariance_matrix = numpy.cov(samples_matrix)
	N = covariance_matrix.shape[0]
	for i in range(N):
		for j in range(i,N):
			covariance_list.append(covariance_matrix[i,j])

	covariance = numpy.array(covariance_list)

	return covariance

def d1_square_d2_square():

	d1_square = []
	d2_square = []
	GRID_CENTERS = grid_cell_centers()
	N = len(GRID_CENTERS)
	for i in range(N):
		for j in range(i,N):
			center_i = GRID_CENTERS[i]
			center_j = GRID_CENTERS[j]
			d1_square.append((center_i[0] - center_j[0])**2.0)
			d2_square.append((center_j[1] - center_i[1])**2.0)

	return d1_square, d2_square

def simple_regression(X, Y):

	parameter = numpy.linalg.solve(numpy.dot(X.T, X), numpy.dot(X.T, Y))

	parameter[0] = numpy.exp(parameter[0])
	parameter[1] = 1.0 / parameter[1]
	parameter[2] = 1.0 / parameter[0]

	parameter = [parameter[0], (parameter[1], parameter[2])]

	return parameter

def kernel_regression_estimate(Players, d1_square, d2_square):

	parameters={'on_ball': None, 'off_ball': None}
	num_grids_x, num_grids_y = grid_num_x_y()
	samples_onball_dict = {'off':{}, "ball": {}, "hoop":{} }
	samples_offball_dict = {'off':{}, "ball": {}, "hoop":{}}
	N = len(Players.keys())
	total_grids = num_grids_x * num_grids_y

	samples_onball_dict['off'] = numpy.zeros((total_grids, N))
	samples_offball_dict['off'] = numpy.zeros((total_grids, N))
	samples_onball_dict['ball'] = numpy.zeros((total_grids, N))
	samples_offball_dict['ball'] = numpy.zeros((total_grids, N))
	samples_onball_dict['hoop'] = numpy.zeros((total_grids, N))
	samples_offball_dict['hoop'] = numpy.zeros((total_grids, N))
	
	row = 0
	for i in range(num_grids_x):
		for j in range(num_grids_y):
			for k,player_id in enumerate(Players.keys()):
				on_Gamma = Players[player_id].Gammas[(i,j)]['off_ball']
				off_Gamma = Players[player_id].Gammas[(i,j)]['on_ball']
				samples_onball_dict['off'][row,k]=on_Gamma[0]
				samples_offball_dict['off'][row,k]=off_Gamma[0]
				samples_onball_dict['ball'][row,k]=on_Gamma[1]
				samples_offball_dict['ball'][row,k]=off_Gamma[1]
				samples_onball_dict['hoop'][row,k]=on_Gamma[2]
				samples_offball_dict['hoop'][row,k]=off_Gamma[2]
			
			count += 1

	on_ball_off = covariance_calculation(samples_dict['on_ball']['off'])
	off_ball_off = covariance_calculation(samples_dict['off_ball']['off'])
	on_ball_ball = covariance_calculation(samples_dict['on_ball']['ball'])
	off_ball_ball = covariance_calculation(samples_dict['off_ball']['ball'])
	on_ball_hoop = covariance_calculation(samples_dict['on_ball']['hoop'])
	off_ball_hoop = covariance_calculation(samples_dict['off_ball']['hoop'])

	num_rows = on_ball_off.shape[0]
	x1 = numpy.ones(num_rows)
	x2 = -0.5 * d1_square 
	x3 = -0.5 * d2_square

	X = numpy.vstack((x1, x2, x3)).T
	prarameters_onball = {"off": None, "ball": None, "hoop": None} 
	prarameters_offball = {"off": None, "ball": None, "hoop": None}

	prarameters_onball['off'] = simple_regression(X, on_ball_off)
	prarameters_offball['off'] = simple_regression(X, off_ball_off)
	prarameters_onball['ball'] = simple_regression(X, on_ball_ball)
	prarameters_offball['ball'] = simple_regression(X, off_ball_ball)
	prarameters_onball['hoop'] = simple_regression(X, on_ball_hoop)
	prarameters_offball['hoop'] = simple_regression(X, off_ball_hoop)

	nu_square_onball = [prarameters_onball['off'][0], prarameters_onball['ball'][0], prarameters_onball['hoop'][0]]
	l_square_onball = [prarameters_onball['off'][1], prarameters_onball['ball'][1], prarameters_onball['hoop'][1]]
	nu_square_offball = [prarameters_offball['off'][0], prarameters_offball['ball'][0], prarameters_offball['hoop'][0]]
	l_square_offball = [prarameters_offball['off'][1], prarameters_offball['ball'][1], prarameters_offball['hoop'][1]]
	parameters['on_ball'] = {'nu_square': nu_square_onball, 'l_square': l_square_onball}
	parameters['off_ball'] = {'nu_square': nu_square_offball, 'l_square': l_square_offball}
	
	return parameters 

def timeit(method):
	def timed(*args, **kw):
		ts = time.time()
		result = method(*args, **kw)
		te = time.time()
		print ('%r (%r, %r) %2.2f sec' % \
			  (method.__name__, args, kw, te-ts))
		return result
	return timed

def kernel(nv_sq, l_sq, x, y):
	""" Defines the gaussian kernel function"""
	edist_square = sum(pow(i-j,2)/l for i, j, l in zip(x,y,l_sq))
	return nv_sq*numpy.exp(-0.5*edist_square)

def cov_matrix(nv_sq, l_sq):
	""" OUTPUT: Generates the covariance matrix for the GRID_CENTERS on the court
		INPUT: nv_sq and l_sq corresponds to the parameters of the kernel
	"""
	GRID_CENTERS = grid_cell_centers()
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
	GRID_CENTERS = grid_cell_centers()
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

def get_posterior_params(W, V, M, K_inv):
	"""
	INPUT: 	W, sparse matrix format, same dimension as K_inv
			V, a numpy vector of size K_inv.shape[0]
			M: the mean vector in numpy format
			K_inv: precision matrix in sparse form
	"""
	from scikits.sparse import cholmod
	posterior_precision = K_inv + W
	posterior_precision_factor = cholmod.cholesky(posterior_precision)
	posterior_mean = posterior_precision_factor(V + K_inv*M)
	return (posterior_mean, posterior_precision) 

def get_F_T_projection():
	"""
	INPUT: Takes 
	OUTPUT: U matrix in csr format
	"""
	import scipy.sparse as sp

	GRID_CENTERS = grid_cell_centers()
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
	samples = sample_mult_normal_given_precision(mu, precision, num_of_samples)
	mu_shift = v_bar+ proj_perp_matrix*mu
	for i in range(num_of_samples):
		samples[:,i] = mu_shift + proj_perp_matrix*samples[:,i]
	return samples

def mean_Gamma_all_players(Players, nu_sq_list, l_sq_list, on_or_off):
	K_inv = full_cov_matrix_inverse(nu_sq_list, l_sq_list)
	Gammas_list=[]
	num_grids_x, num_grids_y = grid_num_x_y()
	for player_id in Players.keys():
		Gamma_array = numpy.asarray([])
		Gammas = Players[player_id].Gammas
		for i in range(num_grids_x):
			for j in range(num_grids_y):
				Gamma_array = numpy.concatenate((Gamma_array, Gammas[(i,j)][on_or_off]), axis=0)
				
		Gammas_list.append(Gamma_array)

	Gammas_matrix = numpy.array(Gammas_list)
	mean_Gammas = numpy.mean(Gammas_matrix, axis=1)

	return mean_Gammas, K_inv

def update_Gamma_sampling(Players, sampling_parameters, PROJ_PERP_MATRIX, V_BAR):
	nu_sq_list_onball = sampling_parameters['onball']['nu_square']
	nu_sq_list_offball = sampling_parameters['offball']['nu_square']
	l_sq_list_onball = sampling_parameters['onball']['l_square'] 
	l_sq_list_offball = sampling_parameters['offball']['l_square']

	mean_Gammas_onball, K_inv_onball = mean_Gamma_all_players(Players, nu_sq_list_onball, l_sq_list_onball, 'on_ball')
	mean_Gammas_offball, K_inv_offball = mean_Gamma_all_players(Players, nu_sq_list_offball, l_sq_list_offball, "off_ball")
	
	for player_id in Players.keys():
		player = Players[player_id]
		W_onball, W_offball = player.make_sparse_diagnal_W()
		V_onball, V_offball = player.stack_V()
		
		posterior_mean, posterior_precision = get_posterior_params(W_onball, V_onball, mean_Gammas_onball, K_inv_onball)
		gamma_sample_onball = conditional_sample(posterior_mean, posterior_precision, 
											1, PROJ_PERP_MATRIX, V_BAR)
		player.Gammas = update_player_Gamma(player, gamma_sample_onball, 'on_ball')

		posterior_mean, posterior_precision = get_posterior_params(W_offball, V_offball, mean_Gammas_offball, K_inv_offball)
		gamma_sample_offball = conditional_sample(posterior_mean, posterior_precision, 
											1, PROJ_PERP_MATRIX, V_BAR)
		player.Gammas = update_player_Gamma(player, gamma_sample_offball, 'off_ball')

	return Players

def mesh_x_y_for_heatmap(num_grids_x, num_grids_y):
	
	x = list(numpy.arange(0, X_LIM, GRID_SIZE))
	x.append(X_LIM)
		
	y = list(numpy.arange(0, Y_LIM, GRID_SIZE))
	y.append(Y_LIM)
	
	x_mesh, y_mesh = numpy.meshgrid(x,y)
	
	return x_mesh, y_mesh


if __name__ == '__main__':
	
	sampling_parameters = {'onball': {'nu_square': [10.0, 10.0, 10.0], 'l_square' : [(0.2, 0.2), (0.2, 0.2), (0.2, 0.2)]}, 'offball': {'nu_square' : [10.0, 10.0, 10.0], 'l_square' : [(0.2, 0.2), (0.2, 0.2), (0.2, 0.2)]}}

	PROJ_PERP_MATRIX, V_BAR = get_F_T_projection()
	d1_square, d2_square = d1_square_d2_square()

	f = open("./input_data/common_game_poss_dict.pickle",'rb')
	common_game_poss_dict = pickle.load(f)

	player_id_list = []
	for team_info in common_game_poss_dict.values():
		player_id_list += team_info[1]

	Players={}
	for player_id, player_name, num_poss in player_id_list:
		Players[player_id] = Player(player_id, player_name)

	global_parameter = Shared_Parameter()
	
	pickle_data_name_list=[]
	for data_file in os.listdir("./input_data/"):
		if data_file.split('_')[1] =='Possessions':
			pickle_data_name_list.append(data_file)
			print(data_file)
	print(len(pickle_data_name_list))

	Possessions, total_log_likelihood = parallel_E_step(pickle_data_name_list, Players, global_parameter)
	
	print('Total_log_likelihood is {}'.format(total_log_likelihood))

	player_poss_dict = find_player_possessions(Possessions)
	Players = Calculate_W_V(Possessions, Players, player_poss_dict)
	
	global_parameter.update_rho(Possessions)

	global_parameter.update_sigma_square(Possessions)

	Players = update_Gamma_sampling(Players, sampling_parameters, PROJ_PERP_MATRIX, V_BAR)

	file = open('./intermediate_data/global_parameter.pickle', 'wb')
	pickle.dump(global_parameter, file)

	file = open('./intermediate_data/Players.pickle', 'wb')
	pickle.dump(Players, file)

	'''
	player_poss_dict = find_player_possessions(Possessions)
	Players = Calculate_W_V(Possessions, Players, player_poss_dict)

	global_parameter.update_rho(Possessions)

	global_parameter.update_sigma_square(Possessions)

	Players = update_Gamma_sampling(Players, sampling_parameters, PROJ_PERP_MATRIX, V_BAR)

	file = open('./intermediate_data/Players.pickle', 'wb')
	pickle.dump(Players, file)
	'''
	'''
	total_log_likelihood_old = total_log_likelihood_new * 1.1
	while abs(total_log_likelihood_new - total_log_likelihood_old)/abs(total_log_likelihood_old) > 0.05:

		total_log_likelihood_old = total_log_likelihood_new
		Possessions, total_log_likelihood_new = parallel_E_step_2(Possessions, Players, global_parameter)
		
		sampling_parameters = kernel_regression_estimate(Players, d1_square, d2_square)

		global_parameter.update_rho(Possessions)

		global_parameter.update_sigma_square(Possessions)

		Players = update_Gamma_sampling(Players, sampling_parameters, PROJ_PERP_MATRIX, V_BAR)

	'''
	










