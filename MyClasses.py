import numpy as np
import heapq

class Node():
    def __init__(self, route, route_set, rcm):
        self.route = route
        self.route_set = route_set
        self.rcm = rcm
        self.priority_val = rcm.lower_bound * ((rcm.ncities - len(route)) / rcm.ncities) + np.random.uniform()

    def update_priority(self):
        self.priority_val = self.rcm.lower_bound * ((self.rcm.ncities - len(self.route)) / self.rcm.ncities) + np.random.uniform()



class ReducedCostMatrix():
    def __init__(self, ncities):
        self.ncities = ncities

        #Matrix is set up so that rows are start city and columns are end city
        self.matrix = [[np.inf] * (ncities) for _ in range(ncities)]
        self.lower_bound = 0

    def copy(self):
        new_rcm = ReducedCostMatrix(self.ncities)
        new_rcm.matrix = [row[:] for row in self.matrix]
        new_rcm.lower_bound = self.lower_bound
        return new_rcm

# Wrapper class for heapq
class MyHeap(object):
    def __init__(self): #, key=lambda x:x):
        self._data = []
    
    def key(self, node):
        return node.priority_val

    # Cost is O(logn)
    def insert(self, item):
        assert( type(item) == Node )
        heapq.heappush(self._data, (self.key(item), item))

    # Cost is O(logn)
    def delete_min(self):
        data = heapq.heappop(self._data)
        assert( type(data[1]) == Node )
        return data[1]

    def size(self):
        return len(self._data)

    def print(self):
        print("Priority Queue:")
        for priority, node in self._data:
            r_format = [''.join(['{}'.format(item._name) for item in node.route])]
            print(f"Priority: {priority}, Route: {r_format}")
