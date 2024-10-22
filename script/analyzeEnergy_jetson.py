#!/bin/env python3

import os
import pandas as pd
import matplotlib as mp
import numpy as np
import sys
import re
import seaborn as sns
import matplotlib.pyplot as plt

def readAllFilesPower(root):
	fl = []
	for (root, dirs, files) in os.walk(root):
		#print ("\t DIRS ", root, dirs, files)
		for f in files:
			if 'gpu_0.txt' in f:
				fl.append(root + "/" + f)
		for d in dirs:
			readAllFilesPower(d)
	return fl


def readAllFilesOutput(root):
	fl = []
	for (root, dirs, files) in os.walk(root):
		#print ("\t DIRS ", root, dirs, files)
		for f in files:
			if 'output.txt' in f:
				fl.append(root + "/" + f)
		for d in dirs:
			readAllFilesOutput(d)
	return fl

def breakLines(line, typ):
	pc = re.findall(r'PC_(.*)_MTX', line)
	mtx = re.findall(r'MTX_(.*)_REP', line)
	rep = re.findall(r'REP_(.*)_' + typ, line)
	return [pc, mtx, rep]

def analyzeOutputFileOut(file):
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


def analyzeOutputFilePower(file):
        scNum = re.compile(r'-?\d+.?\d*(?:[Ee][-+]\d+)?')
        Final_results = []
        flag = False
        with open(file) as fp:
                lines = fp.readlines()
                for ln in lines:
                        ln = ln.strip()
                        tln = ln.split(" ")
                        x = 0
                        flag = False    
                        for t in tln:
                            print ("TK", x, t)
                            if("POM_5V_GPU" in t):
                                print ("Found ", x)
                                flag = True
                                continue
                            if flag:
                                vals = t.split("/")
                                Final_results.append(float(vals[0]))
                                flag = False
                                break
        print ("NM ", Final_results)
        return (Final_results)

def createCurves(lst, typ):
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
				(Fr) = analyzeOutputFilePower(f)
				myMap[ex0][ex1] = Fr
	return myMap

def createHashAllPower(lst, typ):
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
				(Fr) = analyzeOutputFilePower(f)
				myMap[ex0][ex1] = Fr
	myAvgMap = {}
	for d1 in myMap:
		if d1 not in myAvgMap:
			myAvgMap[d1] = {}
		for d2 in myMap[d1]:
			if d2 not in myAvgMap[d1]:
				myAvgMap[d1][d2] = []
			ht = np.array(myMap[d1][d2])
			print ("VAL :::", d1, d2, ht)
			avgPow = np.mean(ht)
			minPow = np.min(ht)
			maxPow = np.max(ht)
			#avgIters = 0
			#avgTime = 0
			#ount = 0
			#maxTime = -1
			#minTime = 999999
			#for d3 in myMap[d1][d2]:
				#print ("ddd ", d1, d2, d3, myMap[d1][d2][d3],  myMap[d1][d2][d3])	
				#avgIters += myMap[d1][d2][d3][1]
				#avgTime += myMap[d1][d2][d3][2]
				#count += 1
				#if myMap[d1][d2][d3][2] > maxTime:
				#	maxTime = myMap[d1][d2][d3][2]
				#if myMap[d1][d2][d3][2] < minTime:
				#	minTime = myMap[d1][d2][d3][2]
			#if(count == 0):
			#	print ("Error in number of experiments ", count)
			#	exit(1)
			#avgIters /= count
			#avgTime /= count
			myAvgMap[d1][d2] = [avgPow, minPow, maxPow]
	return myAvgMap

def createHashAllOut(lst, typ):
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
            if( ex2 in myMap[ex0][ex1].keys()):
                print ("Double assignment, error")
                exit(6)
            (Nl, Fr) = analyzeOutputFileOut(f)
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
                print (d1, d2, d3, myMap[d1][d2][d3])
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
	for d1 in sorted(myMap):
		if(d1 not in tmyMap):
			tmyMap[d1] = {}
		for d2 in sorted(myMap[d1]):
			tmyMap[d1][d2] = myMap[d1][d2][idx]
	return tmyMap

def extractValueIdxPwr(myMap, idx):
    tmyMap = {}
    for d1 in sorted(myMap):
        if(d1 not in tmyMap):
            tmyMap[d1] = {}
        for d2 in sorted(myMap[d1]):
            myarr = np.array(myMap[d1][d2][idx])
            tmyMap[d1][d2] = np.max(myarr)
    return tmyMap

rootDir = sys.argv[1]
extend = sys.argv[2]
print ("Reading the files in ", rootDir)
lst = readAllFilesOutput(rootDir)
#for f in lst:
#	print("File: ", f)

myMap = createHashAllOut(lst, extend)
#for d1 in myMap:
#	print ("K1: ", d1)
#	for d2 in myMap[d1]:
#		print ("\tK2: ", d2)
#		print ("\t\t ", myMap[d1][d2])

#myItersMaps = extractValueIdx(myMap, 0)
myTimeMaps  = extractValueIdx(myMap, 1)
#myMinMaps   = extractValueIdx(myMap, 2)
#myMaxMaps   = extractValueIdx(myMap, 3)

TmDf = pd.DataFrame(myTimeMaps)
TmDf = TmDf.fillna(-1)

lst = readAllFilesPower(rootDir)
myMap = createHashAllPower(lst, extend)

myPowerMaps = extractValueIdxPwr(myMap, 2)
#myPowerMaps = myMap
PwDf = pd.DataFrame(myPowerMaps)
PwDf = PwDf.fillna(-1)
cols = TmDf.index
print(TmDf.index, PwDf.index)
ncols = []
for c in cols:
    ln = c.split("_")
    elem = int(len(ln) / 2)
    if elem == 0:
        elem = 1
    st = ""
    for h in range(elem, len(ln)):
        st = st + ln[h]
    ncols.append(st)

print ("TIME\n")
print (TmDf.values)

print ("PWR\n")
print (PwDf.values)
    
ener = pd.DataFrame(TmDf.values*PwDf.values, columns=TmDf.columns, index=ncols)

ener = ener.div(1000)

ener=ener.replace({0.001000:np.NaN})
print ("Time: ", TmDf )
print ("Power: ", PwDf )
print ("Energy (kJ): ", ener )
plt.figure(figsize = (10,8))
sns.heatmap(ener, cmap = 'Blues_r', center = 0, annot=True, vmax=10, fmt=".2f", square=True, linewidths=.5)

plt.savefig(extend + "_heatmap_v100_" + "power.pdf")
plt.clf()

TmDf.to_csv(extend + "_time_v100.csv")