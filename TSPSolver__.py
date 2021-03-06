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

	def greedy( self,time_allowance=60.0 ):
		results = {}
		nsolutions = 1
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
		end_time = time.time()
		if bssf is not None:
			results['cost'] = bssf.cost
		else:
			results['cost'] = math.inf
		results['time'] = end_time - start_time
		results['count'] = nsolutions
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
		greedy_results = self.greedy(time_allowance)

		self._priority_queue = []
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

		while time.time()-start_time < time_allowance and len(self._priority_queue) > 0:
			node = heapq.heappop(self._priority_queue)[1]
			route = node.route
			route_set = node.route_set
			rcm = node.rcm
			for city in cities:
				if city not in route_set:
					self._states_created += 1
					new_rcm = rcm.copy()
					self.update_rcm(new_rcm, route[-1]._index, city._index, ncities)
					if new_rcm.lower_bound == np.inf:
						self._pruned_states += 1
					else:
						if len(route) == ncities:
							bssf_candidate = TSPSolution(route)
							self._solutions_found += 1
							if bssf_candidate.cost < self._bssf.cost:
								self._bssf = bssf_candidate
							else:
								self._pruned_states += 1
						else:
							new_route = route[:]
							new_route.append(city)
							new_route_set = route_set.copy()
							new_route_set.add(city)
							new_node = Node(new_route, new_route_set, new_rcm)
							heapq.heappush(self._priority_queue, (new_node.priority_val, new_node))
							if len(self._priority_queue) > self._max_queue_size:
								self._max_queue_size = len(self._priority_queue.size())
		end_time = time.time()
		results = {}
		results['cost'] = self._bssf.cost
		results['time'] = end_time - start_time
		results['count'] = self._solutions_found
		results['soln'] = self._bssf
		results['max'] = self._max_queue_size
		results['total'] = self._states_created
		results['pruned'] = self._pruned_states
		return results		


		# Make node for each node not in tree (use set for this?)
		# Use Reduced Cost Matrix to find lower bound on cost
		# add nodes with lower bound less than _rssf pruned to queue (use heapq)
		# If valid path, update bssf

		#rcm class:
			# matrix (n by n) where n is number of cities
			# lower_bound
			# depth


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
						min_cost == rcm.matrix[row][column]
			if min_cost != 0 and min_cost != np.inf:
					rcm.lower_bound += min_cost
					for column in range(ncities):
						rcm.matrix[row][column] -= min_cost
		# Do the same thing for each column
		for column in range(ncities):
			min_cost = np.inf
			for row in range(ncities):
					if rcm.matrix[row][column] < min_cost:
						min_cost == rcm.matrix[row][column]
			if min_cost != 0 and min_cost != np.inf:
					rcm.lower_bound += min_cost
					for row in range(ncities):
						rcm.matrix[row][column] -= min_cost



	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy( self,time_allowance=60.0 ):
		pass
		



