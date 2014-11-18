from __future__ import division
import mincemeat
import os
import forest
from math import log
import numpy
import pickle
from numpy import genfromtxt
import random
import time

my_data = genfromtxt('data/pendigits.train', delimiter=',')
test_data = genfromtxt('data/pendigits.test', delimiter=',')
X = test_data[:,:-1]
y = test_data[:,-1]
def mapfn(dataid,data):
	import forest
	import numpy
	k = 6
	md = 10
	numClass = 10
	train_d = data[0]
	test_d = data[1]
	f = forest.tree(train_d,k,md,numClass)
	f.build(f.root,0)
	probs = f.predict(test_d)
	#predictions = numpy.zeros((len(test_d),numClass))
	blocksize = 500
	numblocks= len(test_d)//blocksize
	print numblocks
	for i in range(0,numblocks):
		blockpred = numpy.zeros((blocksize,numClass))
		for j in range(0,blocksize):
			blockpred[j][numpy.argmax(probs[i*blocksize+j])]+=1
#		print i,blockpred
		yield i,blockpred

def reducefn(key,pred):
	import numpy
	result = 0
	blocksize = 500
	blockpred=pred
	numTrees = 2
	numClasses = 10
	#print key,'\n\n',pred
	#len(blockpred)
	#print len(pred),len(pred[0]),len(pred[0][0])
	#print key
	#if key==6:
		#print pred
		
		#it is of tree x example x classes
		#so prediction to sum is after indexing twice, we get vector of length numclasses
		#
	#blockClassperTree = numpy.zeros((numTrees,blocksize))
	blockClassFinal = numpy.zeros((blocksize))
	#print blockpred,blockpred[0],blockpred[0][0]
	for j in range(0,blocksize):
		#for each example
		for i in range(0,numTrees):
			if i==0:
				sumarr = blockpred[0][j]
			else:
				sumarr = numpy.add(sumarr,blockpred[i][j])
		blockClassFinal[j] = numpy.argmax(sumarr)#assign argmax here
		#blockClassperTree[i][j]=numpy.argmax(sumarr)
		#blcockClassFinal[j]
	#print blockClassperTree
	return blockClassFinal

s = time.time()
bagged = {}
for i in range(0,10):
	ar = numpy.arange(len(my_data))
	n = len(ar)
	bagged[i] = (my_data[numpy.random.choice(ar, n)],test_data)


serpi = mincemeat.Server()
serpi.datasource = bagged
serpi.mapfn = mapfn
serpi.reducefn = reducefn
results = serpi.run_server(password="changeme")
t =time.time()
print 'i think this time is wrong, check with clock. time is: ',t-s
#print results

#gave 