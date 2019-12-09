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
    def __init__( self, difficulty="Easy", npoints = 5, ntests = 5 ):
        self.diff = difficulty
        self.npoints = npoints
        self.ntests = ntests

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

    def start(self):
        self.results = []
        SCALE = 1.0
        self.data_range	= { 'x':[-1.5*SCALE,1.5*SCALE], \
								'y':[-SCALE,SCALE] }
        for i in range(self.ntests):
            print("TEST: " + str(i))  
            self.curSeed = random.randint(0,400)
            points = self.newPoints() # uses current rand seed
            rand_seed = self.curSeed
            self._scenario = Scenario( city_locations=points, difficulty=self.diff, rand_seed=rand_seed )
            self.genParams = {'size':self.npoints,'seed':self.curSeed,'diff':self.diff}
            self.solver = TSPSolver( None )
            self.solver.setupWithScenario(self._scenario)
            self.results.append(self.solver.fancy())
        print(self.results)
        return self.results

print(sys.argv)
if (len(sys.argv) > 1):
    difficulty, npoints, ntests = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    test = AutomatedTester(difficulty, npoints, ntests)
else:
    test = AutomatedTester()
test.start()
# if __name__ == '__main__':
#     	# This line allows CNTL-C in the terminal to kill the program
# 	signal.signal(signal.SIGINT, signal.SIG_DFL)
	
# 	app = QApplication(sys.argv)
# 	w = AutomatedTester()
# 	sys.exit(app.exec())
