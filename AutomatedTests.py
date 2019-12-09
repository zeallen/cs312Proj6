#!/usr/local/bin/python3.7

import math
import random
import signal
import sys

from TSPSolver import *
from TSPClasses import *

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtWidgets import *
	from PyQt5.QtGui import *
	from PyQt5.QtCore import *
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtGui import *
	from PyQt4.QtCore import *
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

class AutomatedTester( ):
    # difficulties are Easy, Normal, Hard, Hard (Deterministic)
    def __init__( self, difficulty="Easy", npoints = 5, ntests = 5, timeout = 10 ):
        self.diff = difficulty
        self.npoints = npoints
        self.ntests = ntests
        self.timeout = timeout

    def newPoints(self):		
        seed = self.curSeed
        random.seed( seed )

        ptlist = []
        RANGE = self.data_range
        xr = self.data_range['x']
        yr = self.data_range['y']
        while len(ptlist) < self.npoints:
            x = random.uniform(0.0,1.0)
            y = random.uniform(0.0,1.0)
            if True:
                xval = xr[0] + (xr[1]-xr[0])*x
                yval = yr[0] + (yr[1]-yr[0])*y
                ptlist.append( QPointF(xval,yval) )
        return ptlist

    def start(self, testType = "fancy"):
        print("***************************\nTEST TYPE: " + testType + "\n**************************")
        self.results = []
        SCALE = 1.0
        self.data_range	= { 'x':[-1.5*SCALE,1.5*SCALE], \
								'y':[-SCALE,SCALE] }
        for i in range(self.ntests):
            self.curSeed = random.randint(0,400)
            points = self.newPoints() # uses current rand seed
            rand_seed = self.curSeed
            self._scenario = Scenario( city_locations=points, difficulty=self.diff, rand_seed=rand_seed )
            self.genParams = {'size':self.npoints,'seed':self.curSeed,'diff':self.diff}
            self.solver = TSPSolver( None )
            self.solver.setupWithScenario(self._scenario)
            self.results.append(getattr(self.solver, testType)(self.timeout))
            # print("TEST: " + str(i), self.results[i])
            # self.results.append(self.solver.fancy())
        i = 0
        avgTime = float(sum(d['time'] for d in self.results)) / len(self.results)
        avgLength = float(sum(d['cost'] for d in self.results)) / len(self.results)
        print("\nAverage Time: ", avgTime)
        print("Average Length: ", avgLength)
        return avgLength

# run all tests with 15 cities: python AutomatedTests final 15
if (len(sys.argv) > 1 and sys.argv[1] == "final"):
    npoints = int(sys.argv[2]) if (len(sys.argv) > 2) else 15
    tests = ["defaultRandomTour", "greedy", "branchAndBound", "fancy"]
    test = AutomatedTester("Hard", npoints, 5, 600)
    randomLength = test.start(tests[0])
    greedyLength = test.start(tests[1])
    print("% of Random: ", float(greedyLength / randomLength), "\n")
    bbLength = test.start(tests[2])
    print("% of Greedy: ", float(bbLength / greedyLength), "\n")
    fancyLength = test.start(tests[3])
    print("% of Greedy: ", float(fancyLength / greedyLength), "\n")
# run 5 B&B easy tests with 10 cities (60 sec): python AutomatedTests branchAndBound Easy 10 5 60
else:
    testType = sys.argv[1] if (len(sys.argv) > 1) else "fancy" 
    difficulty = sys.argv[2] if (len(sys.argv) > 2) else "Hard"
    npoints = int(sys.argv[3]) if (len(sys.argv) > 3) else 15 
    ntests = int(sys.argv[4]) if (len(sys.argv) > 4) else 5
    timeout = int(sys.argv[5]) if (len(sys.argv) > 5) else (600) # 10 minutes
    test = AutomatedTester(difficulty, npoints, ntests, timeout)
    test.start(testType)