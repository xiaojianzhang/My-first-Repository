#!/usr/bin/env python
# encoding: utf-8
r"""
Module containing all functions for data preprocessing.

:Author:
    Sheng Zhang -- Initial version (Aug 2016)
"""

import numpy
import os
import csv
import lxml.etree as ET
import scipy
import pickle
import multiprocessing
import pdb
from class_objects import Possession

season_year = "2015-2016"  #We used data from 2015-2016 NBA season.
hoop_location = numpy.array([5.25, 25.0])  #Hoop Cordinate

def find_pass_possession_moment_player(possession):
	'''
		The function is used to 

	'''
	pass_moment_player = []
	poss_moment_player = []
	quarter = possession.quarter #scalar
	game_clock = possession.game_clock #array
	gamecode = possession.gamecode
	poss_index = possession.poss_index
	file_path = "./NBA_EVENNBA_FINAL_SEQUENCE_PBP_OPTICAL/NBA_FINAL_SEQUENCE_PBP_OPTICAL$"\
				+ gamecode + ".XML"
	root = ET.parse(file_path).getroot()
	
	#figure out the shot_time_game_clock
	shot_time_game_clock = game_clock[-1]
	if "." not in shot_time_game_clock: #in case game-clock = integer
		shot_time_game_clock = shot_time_game_clock + ".00"
	if len(shot_time_game_clock.split(".")[1]) < 2: #in case game-clock is not two decimal precision
		shot_time_game_clock = shot_time_game_clock + "0"
		
	start_of_possession = game_clock[0]
	
	ele = root.find(".//sequence-pbp[@period='{}']/moment[@game-clock='{}']".format\
			  (quarter, shot_time_game_clock)) #starting from the shot time
	player_id = ele.attrib['global-player-id'] #player who takes the shot
	shooter_id = ele.attrib['global-player-id'] #shooter-id
	#event_id = ele.attrib['event-id']
	
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
		if ele is None: #means no previous element 
			break
		else:
			previous_game_clock = float(ele.attrib['game-clock'])


	if len(poss_moment_player) == 0: #if no pass happens within the possession, ball could be in the air before the only ball-handler gets the ball
		poss_moment_player.append((start_of_possession, shooter_id))
		pass_moment_player=[]
	
	return pass_moment_player, poss_moment_player

def identify_ball_states(possession):
	'''
	
	'''
	poss_period = []
	pass_moment_player, poss_moment_player = find_pass_possession_moment_player(possession)
	game_clock = possession.game_clock
	
	if len(pass_moment_player) == 0 and len(poss_moment_player) == 1:#if no pass happens within the possession
		if float(game_clock[0]) > float(poss_moment_player[0][0]):
			poss_period.append((float(poss_moment_player[0][0]), float(game_clock[-1]), poss_moment_player[0][1]))
		
		else:
			poss_period.append((float(game_clock[0]), float(game_clock[-1]), poss_moment_player[0][1]))
	
	else:
		
		if float(pass_moment_player[0][0]) > float(poss_moment_player[0][0]): #the first pass happens before the first poss occurs
			if float(pass_moment_player[0][0]) < float(game_clock[0]):
				poss_period.append((float(game_clock[0]), float(pass_moment_player[0][0]), pass_moment_player[0][1]))

			for i in range(len(pass_moment_player)-1):
				poss_period.append((float(poss_moment_player[i][0]), float(pass_moment_player[i+1][0]),\
									poss_moment_player[i][1]))

			poss_period.append((float(poss_moment_player[-1][0]), float(game_clock[-1]), poss_moment_player[-1][1]))
		
		elif float(pass_moment_player[0][0]) < float(poss_moment_player[0][0]):
			
			for i in range(len(poss_moment_player)-1):
				poss_period.append((float(poss_moment_player[i][0]), float(pass_moment_player[i][0]),\
									poss_moment_player[i][1]))

			poss_period.append((float(poss_moment_player[-1][0]), float(game_clock[-1]), poss_moment_player[-1][1]))
	
	return poss_period

def where_is_the_ball(game_clock, poss_period, off_players):
	for poss in poss_period:
		if game_clock <= poss[0] and game_clock >= poss[1]:
			index = off_players.index(poss[2])
			return index
	 
	return 5 #integer 5 represent that ball in the air at the moment.

def find_ball_states(possession):
	poss_period = identify_ball_states(possession)
	game_clock = possession.game_clock
	num_seq = len(game_clock)
	off_players = possession.offensive_players
	ball_states = numpy.zeros(num_seq, dtype=numpy.int)
	for i in range(num_seq):
		ball_states[i] = where_is_the_ball(float(game_clock[i]), poss_period, off_players)
	
	return ball_states

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

def collecting_possession_data_by_team(team, poss_gameid_num_list):
	
	possession_list=[]#Possession class list
	for gamecode, poss_num in poss_gameid_num_list:
		quarter, game_clock, offensive_team, defensive_team, offensive_players, defensive_players,\
		ball_location, offensive_locations, defensive_locations = get_data_poss_based(gamecode, str(poss_num), season_year)
		possession = Possession(gamecode, str(poss_num), quarter, game_clock, offensive_team, defensive_team, offensive_players,\
		defensive_players, ball_location, offensive_locations, defensive_locations)

		try:
			possession.ball_states = find_ball_states(possession)
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
		file = open("./input_data/{}_Possessions_.pickle".format(team),'wb')
		pickle.dump(Possessions, file)

if __name__ == "__main__":
	f = open("common_game_poss_dict.pickle",'rb')
	common_game_poss_dict = pickle.load(f)
	parallely_collecting_possession_data(common_game_poss_dict)
