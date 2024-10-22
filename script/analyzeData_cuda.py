#!/bin/env Python3

import os
import pandas as pd
import matplotlib as mp
import numpy as np
import sys
import re

def readAllFiles(root):
	fl = []
	for (root, dirs, files) in os.walk(root):
		#print ("\t DIRS ", root, dirs, files)
		for f in files:
			if '.txt' in f:
				fl.append(root + "/" + f)
		for d in dirs:
			readAllFiles(d)
	return fl

def breakLines(line, typ):
	pc = re.findall(r'PC_(.*)_MTX', line)
	mtx = re.findall(r'MTX_(.*)_REP', line)
	rep = re.findall(r'REP_(.*)_' + typ, line)
	return [pc, mtx, rep]

def analyzeOutputFile(file):

        scNum = re.compile(r'-?\d+.?\d*(?:[Ee][-+]\d+)?')
        Final_results = []
        NormList = []
        with open(file) as fp:
                lines = fp.readlines()
                for ln in lines:
                        ln = ln.strip()
                        nms = re.findall(scNum, ln)
                        if ("res norm" in ln):
                                NormList.append(float(nms[-1]))
                        if("Iters" in ln):
                                Final_results.append(float(nms[-1]))
                        elif ("Time" in ln):
                                Final_results.append(float(nms[-1]))
                        elif ("Preconditioner" in ln):
                                fgh = ln.split(" ")
                                Final_results.append(fgh[-1])
        return (NormList, Final_results)

def createHashAll(lst, typ):
	myMap = {}
	for f in lst:
		if('output.txt' in f and typ in f):
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
			(Nl, Fr) = analyzeOutputFile(f)
			myMap[ex0][ex1][ex2] = Fr
	myAvgMap = {}
	for d1 in myMap:
		if d1 not in myAvgMap:
			myAvgMap[d1] = {}
		for d2 in myMap[d1]:
			if d2 not in myAvgMap[d1]:
				myAvgMap[d1][d2] = []
			avgIters = 0
			avgTime = 0
			count = 0
			maxTime = -1
			minTime = 999999
			for d3 in myMap[d1][d2]:
				print ("ddd ", d1, d2, d3, myMap[d1][d2][d3],  myMap[d1][d2][d3])	
				avgIters += myMap[d1][d2][d3][1]
				avgTime += myMap[d1][d2][d3][2]
				count += 1
				if myMap[d1][d2][d3][2] > maxTime:
					maxTime = myMap[d1][d2][d3][2]
				if myMap[d1][d2][d3][2] < minTime:
					minTime = myMap[d1][d2][d3][2]
			if(count == 0):
				print ("Error in number of experiments ", count)
				exit(1)
			avgIters /= count
			avgTime /= count
			myAvgMap[d1][d2] = [avgIters, avgTime, minTime, maxTime]
	return myAvgMap

def extractValueIdx(myMap, idx):
	tmyMap = {}
	for d1 in myMap:
		if(d1 not in tmyMap):
			tmyMap[d1] = {}
		for d2 in myMap[d1]:
			tmyMap[d1][d2] = myMap[d1][d2][idx]
	return tmyMap;

rootDir = sys.argv[1]
extend = sys.argv[2]
print ("Reading the files in ", rootDir)
lst = readAllFiles(rootDir)
#for f in lst:
#	print("File: ", f)

myMap = createHashAll(lst, extend)
for d1 in myMap:
	print ("K1: ", d1)
	for d2 in myMap[d1]:
		print ("\tK2: ", d2)
		print ("\t\t ", myMap[d1][d2])

myItersMaps = extractValueIdx(myMap, 0)
myTimeMaps  = extractValueIdx(myMap, 1)
myMinMaps   = extractValueIdx(myMap, 2)
myMaxMaps   = extractValueIdx(myMap, 3)

ItDf = pd.DataFrame(myItersMaps)
TmDf = pd.DataFrame(myTimeMaps)
MnDf = pd.DataFrame(myMinMaps)
MxDf = pd.DataFrame(myMaxMaps)

ItDf = ItDf.fillna(-1)
TmDf = TmDf.fillna(-1)
MnDf = MnDf.fillna(-1)
MxDf = MxDf.fillna(-1)

print ("Iters\n", ItDf)
print ("Time\n", TmDf)
print ("Min\n", MnDf)
print ("Max\n", MxDf)

ItDf.to_csv("./iters_" + extend + ".csv")
TmDf.to_csv("./time_" + extend + ".csv")
MnDf.to_csv("./min_" + extend + ".csv")
MxDf.to_csv("./max_" + extend + ".csv")
