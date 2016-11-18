import numpy
import os
import csv
import lxml.etree as ET
import multiprocessing
import copy
import pickle

def find_team_gamecodes_season(season_year):
	#this finction will return a dictionary with key:team_name and value:list of gamecodes that team plays
	team_gamecodes_dict={}
	for gamefile in os.listdir("./possessions_sequence_optical/"+season_year+"/"):
		if gamefile.endswith(".csv"):
			#e.g., 2015102701-Atl-Det.csv (possession based sequence optical csv file)
			team1 = gamefile.split("-")[1]
			team2 = gamefile.split("-")[2].split(".")[0]
			gamecode = gamefile[:10]
			
			if team1 not in team_gamecodes_dict.keys(): #if team1 have not appeared 
				team_gamecodes_dict[team1] = [gamecode]
			else:# if team1 have appeared
				team_gamecodes_dict[team1].append(gamecode)

			if team2 not in team_gamecodes_dict.keys(): #if team2 have not appeared 
				team_gamecodes_dict[team2] = [gamecode]
			else:# if team2 have appeared
				team_gamecodes_dict[team2].append(gamecode)
							
	return team_gamecodes_dict

def find_team_players_season(season_year):
	#this finction will return a dictionary with key:team_name and value:list of tuples (player_id, player_name)

	team_players_dict={}
	file_name = "".join(("Players in each team ", season_year, ".csv"))
	f = open(file_name, "r")
	csv_f = csv.reader(f, quotechar='"', delimiter = ',')
	for row in csv_f:
		team = row[0]
		player_name = row[1]
		player_id = row[2]

		if team not in team_players_dict.keys(): #if team have not appeared
			team_players_dict[team] = [(player_id, player_name)]
		else:#if team have appeared
			team_players_dict[team].append((player_id, player_name))
							
	return team_players_dict


def find_player_numofposs_gameid_possindex(team, gamecode_list, player_list, season_year):
	#this function is used to find (gamecode, possessions list) for each given player
	#gamecode_list == team_gamecodes_dict[team]
	#player = (player_id, player_name)
	player_ids_list = [player_id[0] for player_id in player_list]
	#find two team players' id list
	numofposs_gameposslist_dict={} #e.g., numofposs_gameposslist[player_id] = [total_num_of_poss, [(gamecode, poss_num)])

	for gamecode in gamecode_list:
		for game_file in os.listdir("./possessions_sequence_optical/"+season_year+"/"):
			if game_file[:10] == gamecode:
				game_file_path = "./possessions_sequence_optical/"+season_year+"/"+game_file
		
		poss_index_old = '0' 
		f = open(game_file_path, "r")
		csv_f = csv.reader(f, quotechar='"', delimiter = ',')
		for row in csv_f:
			if row[3] != poss_index_old and row[5] == team:
				poss_index = row[3]
				for off_player in row[6:11]:
					id = off_player.strip("( )").split(",")[0].strip(" ")
					if id == "" or id == " ":
						raise ValueError('invalid id')

					if id not in numofposs_gameposslist_dict.keys():
						#find out the index of id in player_list
						index = player_ids_list.index(id)

						numofposs_gameposslist_dict[id] = [player_list[index][1], 1, [(gamecode, poss_index)]]
					
					else:
						numofposs_gameposslist_dict[id][1] += 1
						numofposs_gameposslist_dict[id][2].append((gamecode, poss_index))
					
				poss_index_old = poss_index


	return team, numofposs_gameposslist_dict

def parallel_find_numofposs_gameid_possindex(team_gamecodes_dict, team_players_dict, season_year):

	pool = multiprocessing.Pool(processes=30)
	results = [pool.apply_async(find_player_numofposs_gameid_possindex,\
			  args=(team, team_gamecodes_dict[team], team_players_dict[team], season_year)) for team in team_gamecodes_dict.keys()]

	pool.close()
	pool.join()
	output = [p.get() for p in results]
	numofposs_gameposslist_dict={}
	for result in output:
		team = result[0]
		numofposs_gameposslist_dict[team] = result[1]


	return numofposs_gameposslist_dict 

def select_players_in_each_team_given_condition(numofposs_gameposslist_dict, num_poss_limit):

	numofposs_gameposslist_dict_new = copy.deepcopy(numofposs_gameposslist_dict)

	for team in numofposs_gameposslist_dict_new.keys():
		for player_id, player_info in numofposs_gameposslist_dict[team].items():
			if player_info[1] < num_poss_limit:
				numofposs_gameposslist_dict_new[team].pop(player_id)

	return numofposs_gameposslist_dict_new

def find_common_game_poss(numofposs_gameposslist_dict_new):
	common_game_poss_dict = {}
	#e.g., common_game_poss_dict['team'] = (num_commom_poss, [(playerid,player_name,num_poss)], common_gamecodes_list)
	
	for team in numofposs_gameposslist_dict_new.keys():
		common_game_poss_list = []
		player_numofposs_gameposslist = numofposs_gameposslist_dict_new[team]
		num_commom_poss = 0
		
		players = {}
		for id in player_numofposs_gameposslist.keys():
			name = player_numofposs_gameposslist[id][0]
			players[id] = [id, name, 0]

		player_id_list = [player_id for player_id in player_numofposs_gameposslist.keys()]
		rest_player_id_list = [player_id for player_id in player_numofposs_gameposslist.keys()]

		for player_id in player_id_list:
			if len(rest_player_id_list) <= 4:
				break
			
			else:
				game_poss_list = player_numofposs_gameposslist[player_id][2] #e.g., list[(gamecode, poss_index)]
				rest_player_id_list.remove(player_id)
				for game_poss in game_poss_list:
					count = 1
					common_player_id = []
					for rest_player in rest_player_id_list:
						if game_poss in player_numofposs_gameposslist[rest_player][2]:
							common_player_id.append(rest_player)
							count += 1
						
					if count == 5:
						players[player_id][2] += 1
						for common_player in common_player_id:
							players[common_player][2] += 1
							player_numofposs_gameposslist[common_player][2].remove(game_poss)
						
						num_commom_poss += 1
						common_game_poss_list.append(game_poss)

		players_info_list=[]
		for player_info in players.values():
			players_info_list.append((player_info[0],player_info[1],player_info[2]))

		common_game_poss_dict[team] = [num_commom_poss, players_info_list, common_game_poss_list]
	
	return common_game_poss_dict


if __name__ == '__main__':

	season_year = "2015-2016"
	num_poss_limit = 800

	team_gamecodes = find_team_gamecodes_season(season_year)
	team_players= find_team_players_season(season_year)
	numofposs_gameposslist_dict = parallel_find_numofposs_gameid_possindex(team_gamecodes, team_players, season_year)
	numofposs_gameposslist_dict_new = select_players_in_each_team_given_condition(numofposs_gameposslist_dict, num_poss_limit)
	numofposs_gameposslist_dict_new_copy = copy.deepcopy(numofposs_gameposslist_dict_new)
	common_game_poss_dict = find_common_game_poss(numofposs_gameposslist_dict_new_copy)

	file = open('common_game_poss_dict.pickle', 'wb')
	pickle.dump(common_game_poss_dict, file)
