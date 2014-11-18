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
	predictions = numpy.zeros((len(test_d),numClass))
	blocksize = 500
	numblocks= len(test_d)//blocksize
	for i in range(0,numblocks):
		blockpred = numpy.zeros((blocksize,numClass))
		for j in range(0,blocksize):
			blockpred[j][numpy.argmax(predictions[i*blocksize+j])]+=1
		yield i,blockpred

def reducefn(key,pred):
	import numpy
	result = 0
	blocksize = 500
	
	blockClassFinal = numpy.zeros((blocksize))
	print blockpred
	for i in range(0,blocksize):
		for j in range(0,len(blockpred[0])):
			if j==0:
				sumarr = blockpred[i][0]
			else:
				sumarr = numpy.add(sumarr,blockpred[i][j])
		blockClassFinal[i]=numpy.argmax(sumarr)
	return blockClassFinal

bagged = {}
for i in range(0,2):
	ar = numpy.arange(len(my_data))
	n = len(ar)
	bagged[i] = (my_data[numpy.random.choice(ar, n)],test_data)

	
serpi = mincemeat.Server()
serpi.datasource = bagged
serpi.mapfn = mapfn
serpi.reducefn = reducefn
results = serpi.run_server(password="changeme")


print results