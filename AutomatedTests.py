#!/usr/local/bin/python3.7

import math
import random
import signal
import sys
import csv
from multiprocessing import Process, Queue

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
        self.seeds = []
        for i in range(ntests):
            self.seeds.append(random.randint(0,400))
        # self.curSeed = random.randint(0,400)

    def newPoints(self, curSeed):		
        seed = curSeed
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

    def startNoOutput(self, testType = "fancy"):
        # print("***************************\nTEST TYPE: " + testType + "\n**************************")
        self.results = []
        SCALE = 1.0
        self.data_range	= { 'x':[-1.5*SCALE,1.5*SCALE], \
								'y':[-SCALE,SCALE] }
        for i in range(self.ntests):
            curSeed = self.seeds[i]
            points = self.newPoints(curSeed) # uses current rand seed
            rand_seed = curSeed
            self._scenario = Scenario( city_locations=points, difficulty=self.diff, rand_seed=rand_seed )
            self.genParams = {'size':self.npoints,'seed':curSeed,'diff':self.diff}
            self.solver = TSPSolver( None )
            self.solver.setupWithScenario(self._scenario)

            result = getattr(self.solver, testType)(self.timeout)
            self.results.append(result)
            if result['time'] >= self.timeout:
                break
            print(curSeed)
            # print("TEST: " + str(i), self.results[i])
            # self.results.append(self.solver.fancy())
        i = 0
        avgTime = float(sum(d['time'] for d in self.results)) / len(self.results)
        avgLength = float(sum(d['cost'] for d in self.results)) / len(self.results)
        # print("\nAverage Time: ", avgTime)
        # print("Average Length: ", avgLength)
        return avgTime, avgLength

    def start(self, testType = "fancy"):
        print("***************************\nTEST TYPE: " + testType + "\n**************************")
        avgTime, avgLength = self.startNoOutput(testType)
        print("\nAverage Time: ", avgTime)
        print("Average Length: ", avgLength)
        return avgLength

def runFinal(npoints=15):
    tests = ["defaultRandomTour", "greedy", "branchAndBound", "fancy"]
    test = AutomatedTester("Hard (Deterministic)", npoints, 5, 600)
    #randomLength = test.start(tests[0])
    greedyLength = test.start(tests[1])
    #print("% of Random: ", float(greedyLength / randomLength), "\n")
    #bbLength = test.start(tests[2])
    #print("% of Greedy: ", float(bbLength / greedyLength), "\n")
    fancyLength = test.start(tests[3])
    print("% of Greedy: ", float(fancyLength / greedyLength), "\n")

def getInfoString(testType):
    return "***************************\nTEST TYPE: " + testType + "\n***************************\n"

def getTimeLen(time, length):
    rTime = "Average Time: {}\n".format(time)
    rLen = "Average Length: {}\n".format(length)
    return rTime, rLen

def runFinalMultiprocessed(npoints, q, csvq, numSeconds):
    tests = ["defaultRandomTour", "greedy", "branchAndBound", "fancy"]
    test = AutomatedTester("Hard", npoints, 5, numSeconds)
    randomInfo = getInfoString(tests[0])
    randomTime, randomLength = test.startNoOutput(tests[0])
    rTime, rLen = getTimeLen(randomTime, randomLength)
    greedyTime, greedyLength = test.startNoOutput(tests[1])
    greedyInfo = getInfoString(tests[1])
    gTime, gLen = getTimeLen(greedyTime, greedyLength)
    greedyPercent = 100.0*float(greedyLength / randomLength)
    gPercent = "% of Random: {}\n".format(greedyPercent)
    bbTime, bbLength = test.startNoOutput(tests[2])
    bbInfo = getInfoString(tests[2])
    bTime, bLen = getTimeLen(bbTime, bbLength)
    bbPercent = 100.0*float(bbLength / greedyLength)
    bPercent = "% of Greedy: {}\n".format(bbPercent)
    fancyTime, fancyLength = test.startNoOutput(tests[3])
    fancyInfo = getInfoString(tests[3])
    fTime, fLen = getTimeLen(fancyTime, fancyLength)
    fancyPercent = 100.0*float(fancyLength / greedyLength)
    fPercent = "% of Greedy: {}\n".format(fancyPercent)
    fullOutput = ''.join(["\n\nTest: {}\n\n".format(npoints), randomInfo, rTime, rLen, greedyInfo, gTime, gLen, gPercent ,bbInfo, bTime, bLen, bPercent, fancyInfo, fTime, fLen, fPercent])
    q.put(fullOutput)
    csvq.put([npoints, randomTime, randomLength, greedyTime, greedyLength, greedyPercent, bbTime, bbLength, bbPercent, fancyTime, fancyLength, fancyPercent])

def runSpecific(testType, difficulty, npoints, ntests, timeout):
    test = AutomatedTester(difficulty, npoints, ntests, timeout)
    test.start(testType)


# Auto run all size of tests
if len(sys.argv) == 1:
    numCities = [15, 30, 60, 100, 200, 500, 1000]
    for nCities in numCities:
        runFinal(nCities)

# run all tests multiprocessed: python AutomatedTests all
elif len(sys.argv) == 2:
    # Multiprocessing
    q = Queue()
    csvq = Queue()
    # numCities = [10, 11, 12, 13, 14, 15]
    # numSeconds = 5
    numSeconds = 600
    numCities = [15,30,60,100,200,500,1000]
    processes = []
    for nCities in numCities:
        p = Process(target=runFinalMultiprocessed, args=(nCities, q, csvq, numSeconds))
        p.start()
        processes.append(p)
    with open("results.csv","w+") as csv_file:
        csv_writer = csv.writer(csv_file)            
        csv_writer.writerow(["" , "Random", "Random", "Greedy", "Greedy", "Greedy", "Branch and Bound", "Branch and Bound", "Branch and Bound", "Genetic", "Genetic", "Genetic"])
        csv_writer.writerow(["Num Cities", "Time (sec)", "Path Length", "Time (sec)", "Path Length", "% of Random", "Time (sec)", "Path Length", "% of Greedy", "Time (sec)", "Path Length", "% of Greedy"])
        for p in processes:
            print(q.get())
            csv_writer.writerow(csvq.get())
            p.join()

# run all tests with 15 cities: python AutomatedTests final 15
elif (len(sys.argv) > 1 and sys.argv[1] == "final"):
    npoints = int(sys.argv[2]) if (len(sys.argv) > 2) else 15
    runFinal(npoints)
# run 5 B&B easy tests with 10 cities (60 sec): python AutomatedTests branchAndBound Easy 10 5 60
else:
    testType = sys.argv[1] if (len(sys.argv) > 1) else "fancy" 
    difficulty = sys.argv[2] if (len(sys.argv) > 2) else "Hard"
    npoints = int(sys.argv[3]) if (len(sys.argv) > 3) else 15 
    ntests = int(sys.argv[4]) if (len(sys.argv) > 4) else 5
    timeout = int(sys.argv[5]) if (len(sys.argv) > 5) else (600) # 10 minutes
    runSpecific(testType, difficulty, npoints, ntests, timeout)