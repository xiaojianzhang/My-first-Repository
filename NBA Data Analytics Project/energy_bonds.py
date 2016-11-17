#!/usr/bin/env python
# encoding: utf-8
r"""
Module containing functions for energy bonds model

:Author:
    Sheng Zhang -- Initial version (Sep 2016)
"""
import numpy
import multiprocessing
import scipy

def all_possible_St_Bt_plus_1():
	
	'''
	output:
		    St_Bt_plus_1: list, length == 5**5 * 6 = 18750,
					  all possible combinations of matchups and ball_states. 
	'''

	#[i for i in itertools.product(range(2), repeat=2)] >> [[0, 0], [0, 1], [1, 0], [1, 1]
	St = [list(i) for i in itertools.product(range(5), repeat=5)]
	
	Bt_plus_1 = list(range(6)) #Bt_plus_1 >> [0, 1, 2, 3, 4, 5(means'ball is in_the_air')]

	St_Bt_plus_1 = [i for i in itertools.product(St, Bt_plus_1)]
	#one element of St_Bt_plus_1 is like ([0, 0, 0, 0, 0], 0), i.e., format is (St, Bt_plus_1)

	return St_Bt_plus_1

def all_possible_St_plus_1_Bt_plus_1(St_Bt_plus_1):

	'''
	input:
		   St_Bt_plus_1: tuple, (St, Bt_plus_1), one possible combination of matchups and ball_states.

	output:

		   St_plus_1_Bt_plus_1: list of tuples, length==25, element is like ([0,1,2,3,4], 0, 'changed')
	'''

	matchups = St_Bt_plus_1[0] #St of St_Bt_plus_1, e.g., matchups >> [0,1,2,3,4]
	Bt_plus_1 = St_Bt_plus_1[1] #Bt_plus_1 of St_Bt_plus_1, e.g., Bt_plus_1 >> 0

	changes = [i for i in itertools.product(range(5), repeat=2)] 
	#changes >> [(0, 0), (0, 1), (0, 2), ... , (4,2), (4,3), (4,4)]

	St_plus_1_Bt_plus_1 = []

	for change in changes:
		changed = False #To record whether matchups are changed or not.
		new_matchups = list(matchups) #copy previous matchups for later use
		
		if matchups[change[0]] != change[1]: #if the only defender we can change assignment changed its matchup
			new_matchups[change[0]] = change[1]  #update the matchups
			changed = True

		if changed is True: #if the matchups in the next time step are different than that of previous time step
			St_plus_1_Bt_plus_1.append((new_matchups, Bt_plus_1, 'changed'))
		else:
			St_plus_1_Bt_plus_1.append((new_matchups, Bt_plus_1, 'unchanged'))

	return St_plus_1_Bt_plus_1

def all_possible_St_plus_1(St):

	matchups = St #St of St_Bt_plus_1, e.g., matchups >> [0,1,2,3,4]

	changes = [i for i in itertools.product(range(5), repeat=2)] 
	#changes >> [(0, 0), (0, 1), (0, 2), ... , (4,2), (4,3), (4,4)]

	all_St_plus_1 = []

	for change in changes:

		new_matchups = list(matchups) #copy previous matchups for later use
		
		if matchups[change[0]] != change[1]: #if the only defender we can change assignment changed its matchup
			new_matchups[change[0]] = change[1]  #update the matchups

		all_St_plus_1.append(new_matchups)

	return all_St_plus_1

def state_energy_representation(S_B_tuple):
	'''
	input: 
		    S_B_tuple: tuple in the form of (matchups, ball_states)

	output:
			bonds_energy_array: shape=(4,), array([E_onball_open, E_offball_1on1, E_onball_2on1, E_offball_2on1]).
	'''
	matchups = S_B_tuple[0] #S part of S_B_tuple, e.g., matchups >> [0,1,2,3,4]
	B = S_B_tuple[1] #B part of S_B_tuple, e.g., B >> 0
	state_energy = [0, 0, 0, 0] #corresponding parameters for [E_onball_1on1, E_offball_1on1, E_onball_2on1, E_offball_2on1].
	for offense in range(5):
		num_defense = 0
		for defense in matchups:
			if defense == offense:
				num_defense += 1

		if B == offense: #if the offender has the ball
			
			if num_defense >= 1:
				state_energy[0] += 1	

			if num_defense >= 2:
				state_energy[2] += num_defense - 1
		else:

			if num_defense >= 1:
				state_energy[1] += 1

			if num_defense >= 2:
				state_energy[3] += num_defense - 1

	state_energy_array = numpy.array(state_energy)

	return state_energy_array


def energy_dict_generation(St_Bt_plus_1):

	energy_dict = {}
	#dict_key: tuple(onball_open, num_offball_1on1, num_onball_double, num_offball_double, trans_cost), value: how many times the key occurs
	St_plus_1_Bt_plus_1 = all_possible_St_plus_1_Bt_plus_1(St_Bt_plus_1)
	state_energy_old = state_energy_representation(St_Bt_plus_1)
	for change in St_plus_1_Bt_plus_1:
		S_B_tuple = (change[0],change[1])
		changed_or_unchanged = change[2]
		state_energy_new = state_energy_representation(S_B_tuple)

		if changed_or_unchanged is 'changed':
			trans_cost = -1
		else:
			trans_cost = 0

		key = tuple(list(state_energy_old-state_energy_new) + [trans_cost])

		energy_dict[key] = energy_dict.get(key, 0) + 1

	return energy_dict

def denominator_funvalue_gradient(params, common_energy_dict): #we use
	'''
	input:
          params: 1-D array, shape = (5, ), array([E_onball_open, E_offball_1on1, E_onball_2on1, E_offball_2on1, Trans_cost])

    output:
    	  fun_value: scalar, function form is log(...)
    	  fun_gradient: 1-D array, shape=(4, ), function gradient value

	'''
	num_keys = len(common_energy_dict.keys())
	coeff_matrix = numpy.zeros((num_keys, 5))
	weights = numpy.zeros(num_keys)
	for i,key in enumerate(common_energy_dict.keys()):
		coeff_matrix[i,:] = numpy.array(key)
		weights[i] = common_energy_dict[key]

	intermediate1 = numpy.exp(numpy.dot(coeff_matrix, params))
	intermediate2 = numpy.dot(weights, intermediate1)
	fun_value = numpy.log(intermediate2)

	fun_gradient = numpy.dot(coeff_matrix.T, intermediate1*weights) / intermediate2 #note '*' between array and matrix

	return fun_value, fun_gradient


def St_Bt_plus_1_denominator(params, St_Bt_plus_1):

	denominator_dict = {}	
	for ele in St_Bt_plus_1:
		key = tuple(ele[0] + [ele[1]])
		energy_dict = energy_dict_generation(ele)
		fun_value, fun_gradient = denominator_funvalue_gradient(params, energy_dict)
		denominator_dict[key] = [fun_value, fun_gradient]

	return denominator_dict

def parallel_St_Bt_plus_1_denominator(params):

	denominator_dict = {}	
	St_Bt_plus_1 = all_possible_St_Bt_plus_1()
	St_Bt_plus_1_list=[]
	N = len(St_Bt_plus_1)
	quotient, remainder = divmod(N, 40)
	for i in range(0, 39*quotient, quotient):
		St_Bt_plus_1_list.append(St_Bt_plus_1[i:i+quotient])
		
	St_Bt_plus_1_list.append(St_Bt_plus_1[39*quotient:])

	pool = multiprocessing.Pool(processes=40)
	results = [pool.apply_async(St_Bt_plus_1_denominator,\
			  args=(params, St_Bt_plus_1_list[i])) for i in range(40)]

	pool.close()
	pool.join()
	output = [p.get() for p in results]
	for result in output:
		denominator_dict = {**result, **denominator_dict}

	return denominator_dict

def numerator_funvalue_gradient(params, St_Bt_plus_1, St_plus_1_Bt_plus_1):

	state_energy_t = state_energy_representation(St_Bt_plus_1)
	state_energy_t_plus_1 = state_energy_representation(St_plus_1_Bt_plus_1)
	matchups1 = St_Bt_plus_1[0]
	matchups2 = St_plus_1_Bt_plus_1[0]
	if matchups1 != matchups2:
		coeff = numpy.array(list(state_energy_t - state_energy_t_plus_1) + [-1.0])
		numerator_value = -numpy.dot(coeff, params)
		numerator_gradient = -coeff
	
	else:

		numerator_value = 0
		numerator_gradient = numpy.zeros(5)

	numerator_funval_grad = [numerator_value, numerator_gradient]

	return numerator_funval_grad

def minus_logProb_Gradient_St_St_plus_1(numerator_funval_grad, denominator_funval_grad):

	minus_logProb = numerator_funval_grad[0] + denominator_funval_grad[0]

	Gradient = numerator_funval_grad[1] + denominator_funval_grad[1]

	return minus_logProb, Gradient

def numerator_denominator_funvalue_gradient(params, St_Bt_plus_1, St_plus_1_Bt_plus_1, denominator_dict):

	key = tuple(St_Bt_plus_1[0] + [St_Bt_plus_1[1]])
	denominator_funval_grad = denominator_dict[key]

	numerator_funval_grad = numerator_funvalue_gradient(params, St_Bt_plus_1, St_plus_1_Bt_plus_1)

	return numerator_funval_grad, denominator_funval_grad

def initial_fun_denominator_val_grad(params, B0):

	all_S0 = [list(i) for i in itertools.product(range(5), repeat=5)]
	energy_matrix = numpy.zeros((len(all_S0),4))
	for i,S0 in enumerate(all_S0):
		S0_B0 = (S0, B0)
		energy_array = state_energy_representation(S0_B0)
		energy_matrix[i,:] = energy_array

	denominator1 = numpy.exp(-numpy.dot(energy_matrix, params[:4]))
	denominator2 = numpy.sum(denominator1)
	initial_fun_denom_val = numpy.log(denominator2)

	initial_fun_denom_grad = numpy.dot(-energy_matrix.T, denominator1) / denominator2	

	return initial_fun_denom_val, initial_fun_denom_grad

def initial_fun_numerator_val_grad(params, S0, B0):	

	S0_B0 = (S0, B0)
	energy_array = state_energy_representation(S0_B0)
	init_fun_nume_val = numpy.dot(energy_array, params[:4])
	init_fun_nume_grad = energy_array

	return init_fun_nume_val, init_fun_nume_grad

def initial_fun_val_grad(params, S0, B0):

	init_fun_nume_val, init_fun_nume_grad = initial_fun_numerator_val_grad(params, S0, B0)

	initial_fun_denom_val, initial_fun_denom_grad = initial_fun_denominator_val_grad(params, B0)

	initial_fun_val = init_fun_nume_val + initial_fun_denom_val

	initial_fun_grad = init_fun_nume_grad + initial_fun_denom_grad

	return initial_fun_val, initial_fun_grad

def fun_val_grad(params, Possessions, denominator_dict):

	fun_val = 0
	fun_grad = numpy.zeros(5)
	for i in range(len(Possessions)):
		ball_states = Possessions[i].ball_states
		S_matrix = numpy.array(Possessions[i].sampling_hidden_states_list, dtype=numpy.int)
		for j in range(S_matrix.shape[0]):
			if j == 0:
				S0 = list(S_matrix[0,:])
				B0 = ball_states[0]
				initial_fun_val, initial_fun_grad = initial_fun_val_grad(params, S0, B0)
				fun_val += initial_fun_val
				fun_grad[:4] += initial_fun_grad
			else:
				St = list(S_matrix[j-1,:])
				St_plus_1 = list(S_matrix[j,:])
				Bt_plus_1 = ball_states[j]
				St_Bt_plus_1 = (St, Bt_plus_1)
				St_plus_1_Bt_plus_1 = (St_plus_1, Bt_plus_1)
				numerator_funval_grad, denominator_funval_grad = numerator_denominator_funvalue_gradient(params, St_Bt_plus_1, St_plus_1_Bt_plus_1, denominator_dict)
				minus_logProb, Gradient = minus_logProb_Gradient_St_St_plus_1(numerator_funval_grad, denominator_funval_grad)	
				fun_val += minus_logProb
				fun_grad += Gradient

	return fun_val, fun_grad

def parallel_fun_val_grad(params, Possessions, denominator_dict):

	Possessions_list = []
	length = len(Possessions)
	quotient, remainder = divmod(length, 40)
	for i in range(0, 39*quotient, quotient):
		Possessions_list.append(Possessions[i:i+quotient])
		
	Possessions_list.append(Possessions[39*quotient:])

	pool = multiprocessing.Pool(processes=40)
	results = [pool.apply_async(fun_val_grad,\
			  args=(params, Possessions, denominator_dict)) for Possessions in Possessions_list]

	pool.close()
	pool.join()
	output = [p.get() for p in results]
	total_fun_val = 0
	total_fun_grad = numpy.zeros(5)
	for result in output:
		total_fun_val += result[0]
		total_fun_grad += result[1]

	return total_fun_val, total_fun_grad


def Lipchitz_update(L, y, eta, Possessions):

	p_L = lambda L,y,grad_fy: y - grad_fy / L
	Q_L = lambda L,fy,grad_fy: fy - numpy.dot(grad_fy, grad_fy) / (2.0*L)

	#Find the smallest nonnegative integers i such that F(p_L(y_k)) <= Q_L(p_L(y_k), y_k)
	i = 0.0
	L_new = (eta ** i) * L
	print('L_bar = {}'.format(L_new))
	denominator_dict = parallel_St_Bt_plus_1_denominator(y)
	fy, grad_fy = parallel_fun_val_grad(y, Possessions, denominator_dict) 
	p_L_y = p_L(L_new, y, grad_fy)
	denominator_dict = parallel_St_Bt_plus_1_denominator(p_L_y)
	F_p_L_y, grad_F_p_L_y = parallel_fun_val_grad(p_L_y, Possessions, denominator_dict)
	Q_L_x_y = Q_L(L_new, fy, grad_fy)
	while(F_p_L_y > Q_L_x_y):
		i += 1.0
		L_new = (eta ** i) * L
		print('L_bar = {}'.format(L_new))
		p_L_y = p_L(L_new, y, grad_fy)
		denominator_dict = parallel_St_Bt_plus_1_denominator(p_L_y)
		F_p_L_y, grad_F_p_L_y = parallel_fun_val_grad(p_L_y, Possessions, denominator_dict)
		Q_L_x_y = Q_L(L_new, fy, grad_fy)

	return L_new, grad_fy 

def fista_algorithm(params, Possessions, precision):

	p_L = lambda L,y,grad_fy: y - grad_fy / L

	#step 0:
	L_old = 6e+5 #L_0 > 0
	eta = 1.25 #eta > 1
	x_old = numpy.array(params) #x_0, array
	y_old = x_old #y_1 = x_0, array
	t_old = 1.0 #t1 = 1

	L_new, grad_fy = Lipchitz_update(L_old, y_old, eta, Possessions)
	
	x_new = p_L(L_new, y_old, grad_fy) #array
	print('new energy_params is {}'.format(list(x_new)))
	t_new = (1.0 + numpy.sqrt(1.0 + 4.0 * t_old**2)) / 2.0
	y_new = x_new + ((t_old - 1.0) / t_new) * (x_new - x_old) #array
	denominator_dict = parallel_St_Bt_plus_1_denominator(x_new)
	fun_val, fun_grad = parallel_fun_val_grad(x_new, Possessions, denominator_dict)
	print('f(x) = {}'.format(fun_val))
	print('delta_params = {}'.format(numpy.linalg.norm(x_new - x_old, 2)))

	while numpy.linalg.norm(x_new - x_old, 2) > precision and numpy.linalg.norm(fun_grad, 2) > 1000.0:

		#L_old = L_new
		t_old = t_new
		y_old = y_new #array
		x_old = x_new #array

		#L_new, grad_fy = Lipchitz_update(L_old, y_old, eta, Possessions)
		denominator_dict = parallel_St_Bt_plus_1_denominator(y_old)
		fy, grad_fy = parallel_fun_val_grad(y_old, Possessions, denominator_dict)
		x_new = p_L(L_new, y_old, grad_fy) #array
		print('new energy_params is {}'.format(list(x_new)))
		t_new = (1.0 + numpy.sqrt(1.0 + 4.0 * t_old**2)) / 2.0
		y_new = x_new + ((t_old - 1.0) / t_new) * (x_new - x_old) #array
		denominator_dict = parallel_St_Bt_plus_1_denominator(x_new)
		fun_val, fun_grad = parallel_fun_val_grad(x_new, Possessions, denominator_dict)
		print('f(x) = {}'.format(fun_val))
		print('delta_params = {}, fun_grad_norm = {}'.format(numpy.linalg.norm(x_new - x_old, 2), numpy.linalg.norm(fun_grad, 2)))

	return x_new

if __name__ == "__main__":
	pass