from math import log
import numpy
from numpy import genfromtxt
import random
class node:
	def __init__(self,rows,depth,decisionOf=None, threshold=None, minEx=10):
		self.decisionOf = decisionOf
		self.threshold = threshold
		self.rows = rows
		self.left = None
		self.right = None
		#right means >=
		self.minEx = minEx
		return

class tree:
	def __init__(self,rows, kfeatures, maxDepth):
		self.X = rows[:-1]
		self.y = rows[:1]
		self.root = node(rows, 0)
		self.kfeatures=kfeatures
		self.numFeatures = len(rows[0])-1
		self.maxDepth = maxDepth
		self.nodes=1

 	def build(self, nodeobj, depth):
 		rows = nodeobj.rows
 		if depth>=self.maxDepth:
 			return None
 		if not rows.size:
 			return None
 		a = rows[:,-1]
 		if numpy.all(a == a[0]):
 			return None

 		print self.nodes, nodeobj.rows.shape
 		self.nodes+=1
 		for i in range(0, self.kfeatures):
 			random_index = random.randint(0,self.numFeatures-1)
 			featureValues = {}
 			for row in rows:
 				featureValues[row[random_index]]=1
			 	threshold = get_threshold(featureValues)
 			featureThresholds = {}
 			featureThresholds[random_index] = threshold
		splitBy = pick_feature_split(rows, featureThresholds)
		(set1,set2)=divideSet(rows, splitBy, featureThresholds[splitBy])
		if set1.size:
			nodeobj.left = node(set1, splitBy, featureThresholds[splitBy])
			self.build(nodeobj.left, depth+1)	
		if set2.size:
			nodeobj.right = node(set2, splitBy, featureThresholds[splitBy])
			self.build(nodeobj.right, depth+1)


def pick_feature_split(rows, featureThresholds):
	current_score = entropy(rows)
	best_feature = random.choice(featureThresholds.keys())
	best_gain=0.0

	for feature in featureThresholds.keys():
		threshold = featureThresholds[feature]
		#split set based on the feature and value
		(set1,set2)=divideSet(rows,feature,threshold)
		#Information gain
		p=float(len(set1))/len(rows)
		gain=current_score-p*entropy(set1)-(1-p)*entropy(set2)
		if gain>best_gain and len(set1)>0 and len(set2)>0:
			best_gain=gain
			best_feature = feature
	return best_feature
		#print best_candida

def divideSet(rows, feature, threshold):
	if isinstance(threshold,int) or isinstance(threshold,float):
		split_function = lambda row:row[feature]>=threshold
	else:
		split_function = lambda row:row[feature]==threshold
	set1=[row for row in rows if split_function(row)]
	set2=[row for row in rows if not split_function(row)]
	return numpy.array(set1),numpy.array(set2)

def count(rows):
	rval = {}
	for row in rows:
		#print len(row)
		#print str(int(row[len(row)-1]))
		key = str(int(row[len(row)-1]))
		if row[len(row)-1] not in rval:
			
			rval[key]=1
		else:
			rval[key]+=1
	return rval

def entropy(rows):
	countClasses = count(rows)
	entropy = 0.0
	for c in countClasses.keys():
		prob = float(countClasses[c])/len(rows)
		entropy = entropy - (prob*log(prob))/log(2)
	return entropy

def get_threshold(col_values):
	#todo: make random
	values=col_values.keys()
	values.sort()
	#print len(values)
	#print values
	
	r = values[len(values)/2]
	return r

if __name__ == "__main__":

	numTrees = 100
	my_data = genfromtxt('data/ionosphere.data', delimiter=',')
	newTree = tree(my_data, 6, 10)
	newTree.build(newTree.root,0)