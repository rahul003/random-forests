import forest
import threading
import time
import numpy
from numpy import genfromtxt
from multiprocessing import Queue
# class threadedTree (threading.Thread):
#     def __init__(self, data,kfeatures,numclasses):
#         threading.Thread.__init__(self)

#         self.train_data = data[0]
#         self.test_data = data[1]
#         self.kfeatures = kfeatures
#         self.numClasses = numClasses

#     def run(self):
#         import forest
#         f = forest.tree(self.train_data,self.kfeatures,self.numClasses)
#         f.build(f.root,0)
#         probs = f.predict(self.test_data)
#         return probs
def accuracy(yhat,y):
    correct = 0
    for i in range(0,len(yhat)):
        if yhat[i]==y[i]:
            correct+=1
    print correct, len(yhat)
    return correct/len(yhat)
    # for i in range(0,len(y)):
    #   if probs[i][1]>probs[i][0] and y[i]==1:
    #       correct+=1
    #   elif probs[i][0]>probs[i][1] and y[i]==0:
    #       correct+=1
        #what to do when equal


    #print correct
    #print len(y)
    #print (float)(correct/len(y))
    return correct/len(y)


def ProcessTree(data, i, out_q):
    #print data
    import numpy
    train_data = data[0]
    test_data = data[1]
    kfeatures = data[2]
    numClasses = data[3]
    import forest
    f = forest.tree(train_data,kfeatures,numClasses)
    f.build(f.root,0)
    prob = f.predict(test_data)
    predictions = numpy.zeros((len(test_data),numClasses))
    #print prob[0]
    for i in range(0,len(test_data)):
        maxj = numpy.argmax(prob[i])
        predictions[i][maxj]+=1
    out_q.put(predictions)
    

if __name__=="__main__":
    folder = 'models'
    my_data = genfromtxt('data/pendigits.train', delimiter=',')
    test_data = genfromtxt('data/pendigits.test', delimiter=',')
    
    numTrees = 50
    numClasses = 10
    k = 4
    

    # bagged = {}
    # for i in range(0,numTrees):
    #     ar = numpy.arange(len(my_data))
    #     n = len(ar)
    #     bagged[i] = (my_data[numpy.random.choice(ar, n)],test_data)


    ###thread style
    # threads = []
    # s = time.time()
    # for i in range(0,numTrees):
    #     threads.append(threadedTree(bagged[i],k,numClasses))
    #     threads[i].start()

    # for i in range(0,numTrees):
    #     a = threads[i].join()
    #     print a
    # t= time.time()
    # print 'time',t-s
    # print 'all done'
    ################endofthreadstyle###########
    ####begin process sstyle
    bagged = []
    for i in range(0,numTrees):
        ar = numpy.arange(len(my_data))
        n = len(ar)
        bagged.append((my_data[numpy.random.choice(ar, n)],test_data,k,numClasses))

    import multiprocessing
    # pool = ThreadPool(processes=numTrees)
    # predictions = []

    # async_result = pool.apply_async(ProcessTree,tuple(bagged))
    # predictions.append(async_result.get())
    # print predictions
    jobs = []
    out_q = Queue()
    for i in range(0,numTrees):
        process = multiprocessing.Process(target=ProcessTree,args=(bagged[i],i,out_q))
        jobs.append(process)
    for j in jobs:
        j.start()
    finalPredictions = numpy.zeros((len(test_data),numClasses))
    for i in range(0,numTrees):

        if i==0:
            finalPredictions = out_q.get()
        else:
            tem = out_q.get()
            #print tem
            finalPredictions = numpy.add(finalPredictions,tem)
    yhat = numpy.argmax(finalPredictions,axis=1)
    y = test_data[:,-1]
    print accuracy(yhat,y)
    #print len(yhat)
    for j in jobs:
        j.join()
    print 'done'
   
    #################end
    #treeModels = loadmodels(folder,numTrees)
    #s = time.time()
    #treeModels = train(my_data[:,:],numTrees, numClasses)
    #t = time.time()
    #print 'time',t-s
    
    #allyhats = test(treeModels,test_data[:,:-1],numClasses)
    #yhat = majorityPrediction(allyhats)
    
    
    #print y
    #print accuracy(yhat,y)
    #treeModels[0].predict(my_data[:,])
    #probabilities = test(treeModels)
