#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
from MyClasses import *
import HeapQueue



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def greedy( self,time_allowance=60.0, all_solns = False ):
		results = {}
		solutions = []
		nsolutions = 0
		cities = self._scenario.getCities()
		ncities = len(cities)
		bssf = None
		start_time = time.time()
		for start_city in cities:
			route_set = set()
			route_set.add(start_city)
			route = [start_city]
			current_city = start_city
			while len(route) < ncities and time.time()-start_time < time_allowance:
				min_cost = np.inf
				next_city = None
				for c2 in cities:
					if c2 not in route_set:
						cost_to = current_city.costTo(c2)
						if cost_to < min_cost:
							next_city = c2
							min_cost = cost_to
				if next_city is not None:
					route_set.add(next_city)
					route.append(next_city)
					current_city = next_city
				else:
					break
			if time.time()-start_time < time_allowance and len(route) == ncities:
				bssf_candidate = TSPSolution(route)
				if bssf_candidate.cost < np.inf:
					nsolutions += 1
					#print(f"Cost of solution starting at {start_city._name} - {bssf_candidate.cost}")
					if bssf is None or bssf_candidate.cost < bssf.cost:
						bssf = bssf_candidate
					if all_solns:
						solutions.append(bssf_candidate)

		end_time = time.time()
		if bssf is not None:
			results['cost'] = bssf.cost
		else:
			results['cost'] = math.inf
		results['time'] = end_time - start_time
		results['count'] = nsolutions
		if all_solns:
			results['soln'] = (solutions, bssf)
		else:
			results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results				
	

	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
		
	def branchAndBound( self, time_allowance=60.0 ):
		# Set bssf to greedy solution
		start_time = time.time()
		cities = self._scenario.getCities()
		ncities = len(cities)
		#greedy_results = self.defaultRandomTour(time_allowance)
		greedy_results = self.greedy(time_allowance)
		if greedy_results['soln'] is None:
			greedy_results = self.defaultRandomTour(time_allowance)

		self._priority_queue = MyHeap()
		self._bssf = greedy_results['soln']
		self._states_created = 0
		self._pruned_states = 0
		self._max_queue_size = 0
		self._solutions_found = 0

		#initialize rcm
		start_rcm = self.init_rcm(cities, ncities)
		# Start at first city and create nodes for each city not in route
		route = [cities[0]]
		route_set = set()
		route_set.add(cities[0])
		start_node = Node(route, route_set, start_rcm)
		self._priority_queue.insert(start_node)

		# This creates at most, n! states but won't ever actually make that many
		# Because of this, the total time and space complexity is O(n!n^2)
		while time.time()-start_time < time_allowance and self._priority_queue.size() > 0:
			node = self._priority_queue.delete_min()
			route = node.route
			route_set = node.route_set
			rcm = node.rcm
			if (rcm.lower_bound >= self._bssf.cost):
				self._pruned_states += 1
				continue

			# Runs O(n) times, but the complexity is factored into the while loop
			# The time complexity isn't in the update_rcm function, but is when we make a copy of the matrix
			# So this section is O(n^3)
			for city in cities:
				if city not in route_set:
					self._states_created += 1
					new_route = route[:]
					new_route.append(city)
					new_route_set = route_set.copy()
					new_route_set.add(city)
					new_rcm = rcm.copy()
					self.update_rcm(new_rcm, route[-1]._index, city._index, ncities)
					if new_rcm.lower_bound >= self._bssf.cost:
						self._pruned_states += 1
					else:
						if len(new_route) == ncities:
							bssf_candidate = TSPSolution(new_route)
							self._solutions_found += 1
							if bssf_candidate.cost < self._bssf.cost:
								self._bssf = bssf_candidate
							else:
								self._pruned_states += 1
						else:
							new_node = Node(new_route, new_route_set, new_rcm)
							new_node.update_priority()
							self._priority_queue.insert(new_node)
							if self._priority_queue.size() > self._max_queue_size:
								self._max_queue_size = self._priority_queue.size()
		end_time = time.time()
		if time.time()-start_time < time_allowance:
			self._pruned_states += self._priority_queue.size()
		results = {}
		results['cost'] = self._bssf.cost
		results['time'] = end_time - start_time
		results['count'] = self._solutions_found
		results['soln'] = self._bssf
		results['max'] = self._max_queue_size
		results['total'] = self._states_created
		results['pruned'] = self._pruned_states
		return results		


	# city_start and city_end are indices
	def update_rcm(self, rcm, city_start, city_end, ncities):
		# Add residual cost of edge we ended up taking to lower bound
		rcm.lower_bound += rcm.matrix[city_start][city_end]
		
		# replace every value in that node's rows and columns with np.inf
		for i in range(ncities):
			rcm.matrix[i][city_end] = np.inf
			rcm.matrix[city_start][i] = np.inf 

		# Turn reverse node into infty
		rcm.matrix[city_end][city_start] = np.inf
		
		# reduce so each row and column has a 0 in it (except ones with only infinity)
		self.reduce_rcm(ncities, rcm)


	# O(n^2) time and space complexity
	def init_rcm(self, cities, ncities):
		rcm = ReducedCostMatrix(ncities)
		#fill in edges from graph O(n^2)
		for i in range(ncities):
			for j in range(ncities):
				if i != j:
					rcm.matrix[i][j] = cities[i].costTo(cities[j])
		#self.print_matrix(rcm.matrix)
		# reduce the rcm
		self.reduce_rcm(ncities, rcm)
		return rcm

	# Constant space complexity, O(n^2) time complexity
	def reduce_rcm(self, ncities, rcm):
		# For each row, find lowest number and add to lower bound
		# then replace each node in that row with cost - lowest_cost O(n^2)
		for row in range(ncities):
			min_cost = np.inf
			for column in range(ncities):
					if rcm.matrix[row][column] < min_cost:
						min_cost = rcm.matrix[row][column]
			#print(f"Min cost = {min_cost}")
			if min_cost != 0 and min_cost != np.inf:
					rcm.lower_bound += min_cost
					for column in range(ncities):
						rcm.matrix[row][column] -= min_cost
		# Do the same thing for each column
		for column in range(ncities):
			min_cost = np.inf
			for row in range(ncities):
					if rcm.matrix[row][column] < min_cost:
						min_cost = rcm.matrix[row][column]
			if min_cost != 0 and min_cost != np.inf:
					rcm.lower_bound += min_cost
					for row in range(ncities):
						rcm.matrix[row][column] -= min_cost

	def print_matrix(self, A):
		print('\n'.join([''.join(['{:5} '.format(item) for item in row]) for row in A]))





	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy( self,time_allowance=60.0 ):
		self.population_size = 1000
		self.mating_size = int(self.population_size/2)
		self.num_mutations = int(self.population_size/4)
		self.random_sol_time = 10
		self.greedy_sol_time = 10
		self.total_solutions = 0
		self.bssf_updates = 0
		self.invalid_sols_generated = 0
		self.num_generations = 0
		solution_timeout = 15.0
		self.last_solution_update = time.time()
		start_time = time.time()		
		self.init_population()
		while time.time()-start_time < time_allowance and time.time()-self.last_solution_update < solution_timeout and self.num_generations < 10000:
			# Determine Fitness --> Already done because our population is just the solutions
			# Select mating pool
			mating_population = self.select_mates()
			# Breed
			breeding_order = np.random.permutation(mating_population)
			for i in range(0, len(breeding_order), 2):
				self.breed(breeding_order[i], breeding_order[i+1])
			# Mutate
			for _ in range(self.num_mutations):
				self.mutate(self.population[random.randint(0,len(self.population)-1)])
			# Prune to population size
			self.prune()
			self.num_generations += 1
		end_time = time.time()

		results = {}
		results['cost'] = self.bssf.cost
		results['time'] = end_time - start_time
		results['count'] = self.bssf_updates
		results['soln'] = self.bssf
		results['max'] = self.num_generations
		results['total'] = self.total_solutions
		results['pruned'] = self.invalid_sols_generated
		print(self.bssf_updates)
		return results


	def select_mates(self):
		population_costs = np.array([1/p.cost for p in self.population])
		population_distribution = population_costs/np.sum(population_costs)
		return np.random.choice(self.population, self.mating_size, p=population_distribution)

	def init_population(self):
		# self.population, bssf = [], self.defaultRandomTour()['soln'] 
		self.population, bssf = self.greedy(time_allowance=self.greedy_sol_time, all_solns=True)['soln']
		self.bssf = bssf
		num_iters = 0
		# while len(self.population) < self.population_size or num_iters < self.population_size*5:
		# 	sol = self.defaultRandomTour(time_allowance=self.random_sol_time)['soln']
		# 	self.add_sol(sol)
		# 	num_iters += 1
		while len(self.population) < self.population_size:
			self.add_sol(self.random())

	def mutate(self, sol):
		idx = random.randint(0, len(sol.route)-2)
		route = sol.route.copy()
		route[idx], route[idx+1] = route[idx+1], route[idx]
		new_sol = TSPSolution(route)
		self.add_sol(new_sol)
		

	def add_sol(self, new_sol, keep_inf_prob=.5):
		self.total_solutions += 1
		if new_sol.cost < np.inf or random.random() < keep_inf_prob:
			self.population.append(new_sol)
		elif new_sol.cost == np.inf:
			self.invalid_sols_generated += 1
		if self.bssf is None or new_sol.cost < self.bssf.cost:
				self.bssf = new_sol
				self.last_solution_update = time.time()
				self.bssf_updates += 1
		
	def breed(self, sol1, sol2):
		range1 = random.randint(0, len(sol1.route)-1)
		range2 = random.randint(0, len(sol1.route)-1)

		start_idx = min(range1, range2)
		end_idx = max(range1, range2)
		self.add_sol(self.breed_single(sol1, sol2, start_idx, end_idx))
		self.add_sol(self.breed_single(sol2, sol1, start_idx, end_idx))
	

	def breed_single(self, sol1, sol2, start_idx, end_idx):
		cities = set(map(lambda x: x._index, sol1.route[start_idx:end_idx+1]))
		new_route = sol1.route.copy()
		j = 0
		for i in range(len(sol1.route)):
			if i >= start_idx or i <= end_idx:
				continue
			while sol2.route[j]._index in cities:
				j += 1
			new_route[i] = sol2.route[j]
			j += 1
		return TSPSolution(new_route)

	def prune(self):
		num_to_prune = len(self.population) - self.population_size
		if num_to_prune > 0:
			costs = [p.cost for p in self.population]
			max_cost = max(filter(lambda x: x < np.inf, costs))
			costs = [c if c < np.inf else max_cost for c in costs]
			population_costs = np.array(costs)
			population_distribution = population_costs/np.sum(population_costs)
			delete_routes = np.random.choice(self.population, num_to_prune, p=population_distribution)
			self.population = list(filter(lambda city: city not in delete_routes, self.population))

	def random(self):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		perm = np.random.permutation(ncities)
		route = []
		# Now build the route using the random permutation
		for i in range(ncities):
			route.append(cities[perm[i]])
		bssf = TSPSolution(route)
		return bssf