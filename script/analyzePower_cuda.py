#!/bin/env Python3

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import re

def readAllFiles(root):
	fl = []
	for (root, dirs, files) in os.walk(root):
		#print ("\t DIRS ", root, dirs, files)
		for f in files:
			if 'gpu_0.txt' in f:
				fl.append(root + "/" + f)
		for d in dirs:
			readAllFiles(d)
	return fl

def breakLines(line, typ):
	pc = re.findall(r'PC_(.*)_MTX', line)
	mtx = re.findall(r'MTX_(.*)_REP', line)
	rep = re.findall(r'REP_(.*)_' + typ, line)
	return [pc, mtx, rep]

def createHashAll(lst, typ):
	myMap = {}
	for f in lst:
		if('gpu_0.txt' in f and typ in f):
			ex = breakLines(f, typ)
			ex0 = ex[0][0]
			ex1 = ex[1][0]
			ex2 = ex[2][0]

			if( ex0 not in myMap.keys()):
                        	myMap[ex0] = {}
			if( ex1 not in myMap[ex0].keys()):
				myMap[ex0][ex1] = {}
			if( ex2 in myMap[ex0][ex1].keys() ):
				print ("Double assignment, error")
				exit(6)
			fd = pd.read_csv(f)
			myMap[ex0][ex1][ex2] = fd
	myAvgMap = {}
	for d1 in myMap:
		if d1 not in myAvgMap:
			myAvgMap[d1] = {}
		for d2 in myMap[d1]:
			panels = myMap[d1][d2]
			d3 = list(myMap[d1][d2].keys())
			myAvgMap[d1][d2] = myMap[d1][d2][d3[0]]
			print ("HEAD: ", myAvgMap[d1][d2].columns)
			myAvgMap[d1][d2]['power.draw [W]'] = myAvgMap[d1][d2]['power.draw [W]'].map(lambda x: x.rstrip('W'))
			myAvgMap[d1][d2]['power.draw [W]'] = myAvgMap[d1][d2]['power.draw [W]'].astype(float)
			myAvgMap[d1][d2][' utilization.gpu [%]'] = myAvgMap[d1][d2][' utilization.gpu [%]'].map(lambda x: x.rstrip('%'))
			myAvgMap[d1][d2][' utilization.memory [%]'] = myAvgMap[d1][d2][' utilization.memory [%]'].map(lambda x: x.rstrip('%'))
	return myAvgMap

rootDir = sys.argv[1]
extend = sys.argv[2]
print ("Reading the files in ", rootDir)
lst = readAllFiles(rootDir)
#for f in lst:
#	print("File: ", f)

myMap = createHashAll(lst, extend)
outstr = ", , Mean, Min, Max\n"
for d1 in sorted(myMap):
	#print ("K1: ", d1)
	for d2 in sorted(myMap[d1]):
		#print ("\tK2: ", d2)
		#print ("\t\t ", myMap[d1][d2])
		#plt.plot(myMap[d1][d2]['power.draw [W]'])
		myMap[d1][d2]['power.draw [W]'].plot.line()
		plt.savefig(extend + "_" + d1 + "_" + d2 + "power.pdf")
		plt.clf()
		currPow = np.array(myMap[d1][d2]['power.draw [W]'])
		outstr = outstr + d1 + "," + d2 + "," + str(np.mean(currPow)) + ", " + str(np.min(currPow)) + ", " + str(np.max(currPow)) + "\n"
		print ("Case: " + outstr )
with open(extend + "_power_stats_a100.csv", "w") as fp:
	fp.write(outstr)	
