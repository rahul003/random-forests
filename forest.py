from __future__ import division
from math import log
import numpy
import pickle
from numpy import genfromtxt
import random
import time

class node:
	def __init__(self,rowindices,depth,decisionOf=None, threshold=None):
		self.decisionOf = decisionOf
		self.threshold = threshold
		self.rowindices = rowindices
		self.left = None
		self.right = None
		return

class tree:
	def __init__(self,rows, kfeatures, numClasses, maxDepth=10, minEx=10):
		self.rows = rows
		self.root = node(numpy.arange(0,len(rows)), 0)
		self.kfeatures=kfeatures
		self.numFeatures = len(rows[0])-1
		self.maxDepth = maxDepth
		self.nodes=1
		self.minEx = minEx
		self.numClasses = numClasses

	def getKFeatureThresholds(self,ro):
		featureThresholds = {}
 		for i in range(0, self.kfeatures):
 			
 			random_index = random.randint(0,self.numFeatures-1)
 			#random_index = i

 			featureValues = {}
 			for row in ro:
 				featureValues[row[random_index]]=1
			 	threshold = get_threshold(featureValues)
 			featureThresholds[random_index] = threshold
 		return featureThresholds

 	def build(self, nodeobj, depth):
 		rowindices = nodeobj.rowindices
 		#print depth,len(rowindices)
 		#print self.rows
 		#if cross depth, return
 		if depth>=self.maxDepth:
 			return None

 		#if nodeobj doesnt have any rows, return
 		if not rowindices.size:
 			return None

 		ro = self.rows[rowindices]
 		a = ro[:,-1]
 		
 		#if all class equal, return none
 		if numpy.all(a == a[0]):
 			return None

 		#print self.nodes, depth,nodeobj.rowindices.shape
 		featureThresholds = self.getKFeatureThresholds(ro)
		splitBy = self.pick_feature_split(rowindices, featureThresholds)
		(set1,set2)=self.divideSet(rowindices, splitBy, featureThresholds[splitBy])
		#print set1.size,set2.size
		if set1.size>self.minEx and set2.size>self.minEx:
			nodeobj.decisionOf = splitBy
			nodeobj.threshold = featureThresholds[splitBy]
			nodeobj.left = node(set1, depth+1)
			self.nodes+=1
			self.build(nodeobj.left, depth+1)	
		
			nodeobj.decisionOf = splitBy
			nodeobj.threshold = featureThresholds[splitBy]
			nodeobj.right = node(set2, depth+1)
			self.nodes+=1
			self.build(nodeobj.right, depth+1)

	def divideSet(self,rowindices, feature, threshold):
		#print threshold
		#print self.rows[rowindices]
		if isinstance(threshold,int) or isinstance(threshold,float):
			split_function = lambda row:self.rows[row][feature]>=threshold
		else:
			split_function = lambda row:self.rows[row][feature]==threshold
		set1=[row for row in rowindices if split_function(row)]
		set2=[row for row in rowindices if not split_function(row)]
		#print set1, set2
		#print self.rows[set1], self.rows[set2]
		return numpy.array(set1),numpy.array(set2)

	def count(self,rowindices):
		rval = {}
		for t in rowindices:
			key = str(int(self.rows[t][len(self.rows[0])-1]))
			if key not in rval:
				rval[key]=1
			else:
				rval[key]+=1
		return rval

	def pick_feature_split(self,rowindices, featureThresholds):
		#print featureThresholds
		current_score = self.entropy(rowindices)
		best_feature = random.choice(featureThresholds.keys())
		best_gain=0.0

		for feature in featureThresholds.keys():
			threshold = featureThresholds[feature]
			#split set based on the feature and value
			(set1,set2)=self.divideSet(rowindices,feature,threshold)
			#Information gain
			p=float(len(set1))/len(rowindices)
			gain=current_score-p*self.entropy(set1)-(1-p)*self.entropy(set2)
			if gain>best_gain and len(set1)>0 and len(set2)>0:
				best_gain=gain
				best_feature = feature
			#print gain
		#print 'best',best_gain
		return best_feature
			#print best_candida

	def entropy(self,rows):
		countClasses = self.count(rows)
		entropy = 0.0
		for c in countClasses.keys():
			prob = float(countClasses[c])/len(rows)
			entropy = entropy - (prob*log(prob))/log(2)
		return entropy

	def printtree(self,nodej,indent=''):
  	 # Is this a leaf node?
	   if nodej.left==None and nodej.right==None:
	      print str(len(nodej.rowindices))
	   else:
	      # Print the criteria
	      print len(nodej.rowindices), '; dec ',str(nodej.decisionOf)+':'+str(nodej.threshold)+'? '
	      # Print the branches
	      print indent+'T->',
	      self.printtree(nodej.left,indent+'  ')
	      print indent+'F->',
	      self.printtree(nodej.right,indent+'  ')

	def savemodel(self, filename):
		fp = open(filename, "w")
		pickle.dump(self, fp)
		fp.close()
		return

	def getProb(self, nodeobj):
		countdict = self.count(nodeobj.rowindices)
		#print countdict
		totalcount = 0
		for key in countdict.keys():
			totalcount+=countdict[key]
		prob = []
		for key in countdict.keys():
			prob.append((key,countdict[key]/totalcount))
		#print prob
		return prob

	def traverse(self, node, example):
		prob = None
		#print '\nnewexample:'
		#print node.left, node.right
		while prob is None:
			if node.left is None and node.right is None: #leaf of tree
				#print 'leaf'
				prob = self.getProb(node)
			elif node.left is None:
				#print 'went right'
				#print node.decisionOf,example[node.decisionOf], node.threshold
				#print example
				#print threshold
				if example[node.decisionOf]>=node.threshold:
					node = node.right
				else:
					prob = self.getProb(node)
			elif node.right is None:
				#prin2t 'went left'
				if example[node.decisionOf]<node.threshold:
					node = node.left
				else:
					prob = self.getProb(node)
			else:
				if example[node.decisionOf]>=node.threshold:
					node = node.right
				else:
					node=node.left
		return prob

	def predict(self,X):
		probs = numpy.zeros((len(X),self.numClasses))
		#print 'predicting',len(X)
		for i in range(0,len(X)):
			p = self.traverse(self.root,X[i])
			for j in p:
				#print j
				(a,b) = j
				probs[i][int(a)] = b

		#print probs
		return probs

def get_threshold(col_values):
	#todo: make random
	
	values=col_values.keys()
	#print values
	values.sort()
	#print values
	r = values[len(values)//2]
	#print r,'\n'
	return r

def train(mydata,numTrees,kfeatures,numClasses):
	
	trees = []
	
	for i in range(0,numTrees):
		ar = numpy.arange(len(my_data))
		n = len(ar)
		
		#print n
		#gen random subset of dataexamples of size n
		bagged = numpy.random.choice(ar, n)
		newTree = tree(my_data[bagged],kfeatures,numClasses)
		newTree.build(newTree.root,0)
		print 'saving model',i
		newTree.savemodel('models/'+str(i)+'.txt')
		trees.append(newTree)
	return trees

def test(models,X,numClasses):
	allModelsProbs=numpy.zeros((len(models),len(X),numClasses))
	#print allModelsProbs
	#allModelsProbs = []
	for i in range(0,len(models)):
		#models[i].printtree(models[i].root)
		t = models[i].predict(X)
		for j in range(len(X)):
			for k in range(0,numClasses):
				allModelsProbs[i][j][k]=t[j][k]
	return allModelsProbs

def majorityPrediction(allyhats):
	numTrees = len(allyhats)
	num_test_data = len(allyhats[0])
	numClasses = len(allyhats[0][0])
	numPredictions = numpy.zeros((num_test_data,numClasses))
	yhat = numpy.zeros((num_test_data,))
	#print avgprobs
	for j in range(0,num_test_data):
			for i in range(0,numTrees):
				numPredictions[j][numpy.argmax(allyhats[i][j])]+=1
	#print allyhats
	#print avgprobs
	for j in range(0,num_test_data):
		yhat[j]=numpy.argmax(numPredictions[j])
	#print numPredictions
	#print yhat
	return yhat

def accuracy(yhat,y):
	correct = 0
	for i in range(0,len(yhat)):
		if yhat[i]==y[i]:
			correct+=1
	print correct, len(yhat)
	return correct/len(yhat)
	# for i in range(0,len(y)):
	# 	if probs[i][1]>probs[i][0] and y[i]==1:
	# 		correct+=1
	# 	elif probs[i][0]>probs[i][1] and y[i]==0:
	# 		correct+=1
		#what to do when equal


	#print correct
	#print len(y)
	#print (float)(correct/len(y))
	return correct/len(y)

def loadmodels(directory,numTrees):
	'''
	yet to test
	'''
	trees = []
	for i in range(0,numTrees):
		tree = pickle.load( open( directory+'/'+str(i)+'.txt', "r" ))
		trees.append(tree)
	return trees

if __name__ == "__main__":
	#train or load
	folder = 'models'
	my_data = genfromtxt('data/pendigits.train', delimiter=',')
	#print my_data.shape
	numTrees = 10
	numClasses = 10
	kfeatures = 4
	#treeModels = loadmodels(folder,numTrees)
	#s = time.time()
	treeModels = train(my_data[:,:],numTrees,kfeatures, numClasses)
	#t = time.time()
	#print 'time',t-s
	

	test_data = genfromtxt('data/pendigits.test', delimiter=',')
	allyhats = test(treeModels,test_data[:,:-1],numClasses)
	#print allyhats
	yhat = majorityPrediction(allyhats)
	#print 'gotyhat'
	#print avgprobs
	#y = test_data[:,-1]
	#print y
	#print accuracy(yhat,y)
	
