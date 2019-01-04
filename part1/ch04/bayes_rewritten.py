'''
bayes_rewritten: Naive Bayes All-in-one Rewritten Version

@author: Teddy.Ma
'''

from numpy import *
import operator as op
import os
import pandas as pd

currrentDir = os.path.dirname(os.path.realpath(__file__))

# return a sorted pair list
def pairSorted(pairItems, desc = True):
    return sorted(pairItems, key=lambda pair: pair[1], reverse=desc)

# return a sorted dictionary item list
def dicSorted(dicItems, desc = True):
    return sorted(dicItems, key=op.itemgetter(1), reverse=desc)

# 4.5.1 generate simple test data
def loadDataSet():
    docs=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
           ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
           ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
           ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
           ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
           ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    labels = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return docs, labels

# 4.5.1 create a list of all the distinct base words for training and classify
def createVocabList(docs):
    vocabSet = set([])  #create empty set
    for doc in docs:
        vocabSet = vocabSet | set(doc) #union of the two sets
    return list(vocabSet)

# 4.5.1 normalize a document to the 0/1 value base words based vector
def setOfWords2Vec(vocabList, doc):
    returnVec = [0]*len(vocabList)
    for word in doc:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def createTrainMat(docs, vocabList):
    trainMat=[]
    for doc in docs:
        trainMat.append(setOfWords2Vec(vocabList, doc))
    return trainMat

# 4.5.2 Naïve Bayes classifier training
def trainNB0(trainMat,trainLabels):
    numTrainDocs = len(trainMat)
    numWords = len(trainMat[0])
    pAbusive = sum(trainLabels)/float(numTrainDocs)
    p0Num = ones(numWords); p1Num = ones(numWords)      # 4.5.3 change from zeros to ones() for real world
    p0Denom = 2.0; p1Denom = 2.0                        #4.5.3 change from zeros to 2.0 to for real world
    for i in range(numTrainDocs):
        if trainLabels[i] == 1:
            p1Num += trainMat[i]
            p1Denom += sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    p1Vect = log(p1Num/p1Denom)          # 4.5.3 change to log() to avoid underflow issue
    p0Vect = log(p0Num/p0Denom)          # 4.5.3 change to log() to avoid underflow issue
    return p0Vect,p1Vect,pAbusive

# 4.5.3 classify a document vector with the trained output of trainNB0
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
    
# 4.5.3 test Naïve Bayes classifier
def testingNB():
    docs, labels = loadDataSet()
    vocabList = createVocabList(docs)
    trainMat = createTrainMat(docs, vocabList)
    p0V,p1V,pAb = trainNB0(array(trainMat),array(labels))
    testDoc = ['love', 'my', 'dalmation']
    testVec = array(setOfWords2Vec(vocabList, testDoc))
    print (testDoc,'classified as: ',classifyNB(testVec,p0V,p1V,pAb))
    testDoc = ['stupid', 'garbage']
    testVec = array(setOfWords2Vec(vocabList, testDoc))
    print (testDoc,'classified as: ',classifyNB(testVec,p0V,p1V,pAb))

def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 

# 4.5.4 setOfWords2Vec to consider the weight of word occurence
def bagOfWords2VecMN(vocabList, doc):
    returnVec = [0]*len(vocabList)
    for word in doc:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
    
# 4.6.2 test spam email
def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open(currrentDir + '/email/spam/%d.txt' % i, 'r', encoding='Windows-1252').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open(currrentDir + '/email/ham/%d.txt' % i, 'r', encoding='Windows-1252').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)    #create vocabulary
    trainingSet = list(range(50)); testSet=[]           #create test set
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:    #train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print ("classification error",docList[docIndex])
    print ('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,fullText

# 4.7.1 only lick the top 30 most frequent words
def calcMostFreq(vocabList,fullText):
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = dicSorted(freqDict.items()) 
    return sortedFreq[:30]       

if __name__ == "__main__":
    print("Simple dataset testing:\n")
    testingNB()
    print("\n---\n")
    print("Spam Email testing:\n")
    vocabList,fullText = spamTest()
    print("\nTop 30 most frequent words: ", calcMostFreq(vocabList,fullText))
