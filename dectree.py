from math import log
import numpy
import cPickle
from numpy import genfromtxt
import random
class node:
	def __init__(self,rowindices,depth,decisionOf=None, threshold=None, minEx=10):
		#print decisionOf, threshold
		self.decisionOf = decisionOf
		self.threshold = threshold
		self.rowindices = rowindices
		self.left = None
		self.right = None
		#right means >=
		self.minEx = minEx
		return

class tree:
	def __init__(self,rows, kfeatures, maxDepth):
		self.X = rows[:-1]
		self.y = rows[:1]
		self.rows = rows
		self.root = node(numpy.arange(0,len(rows)), 0)
		self.kfeatures=kfeatures
		self.numFeatures = len(rows[0])-1
		self.maxDepth = maxDepth
		self.nodes=1

 	def build(self, nodeobj, depth):
 		rowindices = nodeobj.rowindices
 		if depth>=self.maxDepth:
 			return None
 		if not rowindices.size:
 			return None
 		ro = self.rows[rowindices]
 		a = ro[:,-1]
 		if numpy.all(a == a[0]):
 			return None

 		#print self.nodes, depth,nodeobj.rowindices.shape
 		self.nodes+=1
 		featureThresholds = {}
 		for i in range(0, self.kfeatures):
 			random_index = random.randint(0,self.numFeatures-1)
 			featureValues = {}
 			for row in ro:
 				featureValues[row[random_index]]=1
			 	threshold = get_threshold(featureValues)
 			featureThresholds[random_index] = threshold
 		#print featureThresholds
		splitBy = self.pick_feature_split(rowindices, featureThresholds)
		#print splitBy
		(set1,set2)=self.divideSet(rowindices, splitBy, featureThresholds[splitBy])
		if set1.size:
			nodeobj.left = node(set1, depth+1,splitBy, featureThresholds[splitBy])
			self.build(nodeobj.left, depth+1)	
		if set2.size:
			nodeobj.right = node(set2, depth+1,splitBy, featureThresholds[splitBy])
			self.build(nodeobj.right, depth+1)

	def divideSet(self,rowindices, feature, threshold):
		#print threshold
		if isinstance(threshold,int) or isinstance(threshold,float):
			split_function = lambda row:self.rows[row][feature]>=threshold
		else:
			split_function = lambda row:self.rows[row][feature]==threshold
		set1=[row for row in rowindices if split_function(row)]
		set2=[row for row in rowindices if not split_function(row)]
		#print set1, set2
		return numpy.array(set1),numpy.array(set2)

	def count(self,rowindices):
		#print rowindices
		rval = {}
		for t in rowindices:
			#print len(row)
			#print str(int(row[len(row)-1]))
			key = str(int(self.rows[t][len(self.rows[0])-1]))
			#print key
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
	      print str(nodej.decisionOf)+':'+str(nodej.threshold)+'? '

	      # Print the branches
	      print indent+'T->',
	      self.printtree(nodej.left,indent+'  ')
	      print indent+'F->',
	      self.printtree(nodej.right,indent+'  ')

	def savemodel(self, filename):
		fp = open(filename, "w")
		cPickle.dump(self, fp)
		fp.close()
		return

	def predict(self,X):
		return

def get_threshold(col_values):
	#todo: make random
	values=col_values.keys()
	values.sort()
	r = values[len(values)/2]
	return r

def train(mydata,numTrees):
	
	kfeatures = 6
	maxdepth = 10
	trees = []
	
	for i in range(0,numTrees):
		ar = numpy.arange(len(my_data))
		n = len(ar)
		
		#print n
		#gen random subset of dataexamples of size n
		bagged = numpy.random.choice(ar, n)
		newTree = tree(my_data[bagged],kfeatures,maxdepth)
		newTree.build(newTree.root,0)
		print 'saving model',i
		newTree.savemodel('models/'+str(i)+'.txt')
		trees.append(newTree)
	return trees

def test(models):
	probs = []
	for mod in models:
		mod.predict(X)
	return probs

def loadmodels(directory,numTrees):
	trees = []
	for i in range(0,100):
		tree = pickle.load( open( directory+'/'+str(i)+'.txt', "r" ))
		trees.append(tree)
	return trees

if __name__ == "__main__":
	#train or load
	folder = 'models'
	my_data = genfromtxt('data/ion-train.data', delimiter=',')
	#treeModels = loadmodels(folder,numTrees=100)
	treeModels = train(my_data,numTrees=1)
	test(treeModels)
	#treeModels[0].predict(my_data[:,])
	#probabilities = test(treeModels)
