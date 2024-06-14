import sys
from .utils import *
import numpy as np
import pandas as pd
from time import process_time
import heapq
from collections import defaultdict
from sklearn.linear_model import RidgeClassifierCV
import time
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from .helper import createDatabase
from sklearn.base import BaseEstimator
class ZTime(BaseEstimator):
    # self.window_size, 
                            # self.window_size_slope,
                            # self.alphabet_size,
                            # self.alphabet_size_slope,
                            # self.glob)
    def __init__(self, step = 20, split_no = 1, window_size = 5, window_size_slope = 1, alphabet_size = 5, alphabet_size_slope = 5,
                 glob = False, random_state = np.random.randint(np.iinfo(np.int32).max), forgettable=True):
        # From Z-Miner
        #
        # F (arrangements information) and L (locations) are practially saved together.
        # F is 1-2 dimension of each level, L is 3-5 dimension for Z-Table and 2-3 dimension for Z-Arrangements
        # F is used as a key of locations as stated in the paper.
        #
        #self.FL = defaultdict(lambda: defaultdict(int))
        # self.tempFL = defaultdict(lambda: defaultdict(set))
        self.FL = defaultdict(lambda: defaultdict(int))
        # For debug: total frequency and comparison count
        self.comparisoncount = 0
        self.totalfrequency = 0
        # For memory garbage collection
        self.forgettable = forgettable
        self.step = step
        self.split_no = split_no
        self.window_size = window_size
        self.window_size_slope = window_size_slope
        self.alphabet_size = alphabet_size
        self.alphabet_size_slope = alphabet_size_slope
        self.glob = glob
        self.random_state = random_state
        # Indicator for test/train mode
        self.isTest = False
        # This variable will keep the event labels kept in the training phase
        self.initialSupports = None



    def pruneTestEventLabels(self):
        for seq in self.database.sequences:
            prunedSequences = []
            for event in seq.sequences:
                print(event, event.label)
                if event.label in self.initialSupports:
                    prunedSequences.append(event)
            seq.sequences = prunedSequences
        return

    # def pruneWithSupport(self):
    #     """
    #     For time seires in this phase we do not consider maxSup as all the events belong
    #     """
    #     copiedEvents = self.database.initialSupport.copy()
    #     # remove event below threshold
    #     for label, support in self.database.initialSupport.items():
    #         if ((support < self.constraints["minSup"])):
    #             del copiedEvents[label]
        
    #     self.initialSupports = copiedEvents
    #     for seq in self.database.sequences:
    #         prunedSequences = []
    #         for event in seq.sequences:
    #             if (
    #                 event.label in self.initialSupports
    #                 and ((self.initialSupports[event.label]
    #                 >= self.constraints["minSup"]))
    #             ):
    #                 prunedSequences.append(event)
    #         seq.sequences = prunedSequences
    #     return

    def createZTable(self):
        # each e-sequence id to generate next
        for sid in range(len(self.database)):
            S = self.database[sid]
            # iterate every event pairs in the same sequence
            # print("Now list ")
            for s1_idx in range(len(S)):
                #for s2_idx in range(s1_idx, len(S.sequences)):
                # if self.constraints["step"] == 0:
                #     max_range = len(S.sequences)
                # else:
                #     max_range = np.min([len(S.sequences), s1_idx + self.constraints["step"]])
                max_range = np.min([len(S), s1_idx + self.step])
                tmp_pair = set()
                for s2_idx in range(s1_idx, max_range):
                
                    s1 = S[s1_idx]
                    s2 = S[s2_idx]
                    # Gap control: if there is no chance to make a pair? we can maybe skip it
                    if self.step == 0:
                       break

                    R2 = getRelation(s1, s2)
                    
                    if R2 != None:

                        E2 = (s1[2], s2[2], R2)
                        # Testing regulation ...
                        if ((self.isTest == False) or ((self.isTest == True) and (E2 in self.trainArrangementsSet))):
                            
                            # initialization
                            # F parts: frequent arrangements
                            # event pair hash table
                            #if (E2, R2) not in self.FL:
                                #self.FL[(E2, R2)] = defaultdict(int)
                            if E2 not in tmp_pair:
                                self.FL[E2][sid] += 1
                                tmp_pair.add(E2)
                                
                            # relation part of frequent arrangements
                            # relation hash table
                            # elif S.id not in self.FL[(E2, R2)]:
                                # self.FL[(E2, R2)][S.id] = 1
                            # else:
                                # self.FL[(E2, R2)][S.id] += 1
                            # L parts: sequence location
                            # e-sequence hash table
                            # elif S.id not in self.FL[2][E2][R2]:
                            #     self.FL[E2][R2][S.id] = {s1: 1}
                            # first interval hash table
                            # elif s1 not in self.FL[2][E2][R2][S.id]:
                                # self.FL[2][E2][R2][S.id][s1] = 0
                            # second interval hash table
                            # adding all addresses of s2 having a relation with s1
                            #self.FL[(E2, R2)][S.id][s1].append(s2)

        # for pair in self.tempFL:
        #     for sid in self.tempFL[pair]:
        #         self.FL[pair][sid] = len(self.tempFL[pair][sid])
        
        # del self.tempFL
        #VERSION 2
        # minSup will not be used anymore
        # for E2 in list(self.FL[2]):
        #     for R2 in list(self.FL[2][E2]):
        #         if len(self.FL[2][E2][R2]) < self.constraints["minSup"]:
        #             del self.FL[2][E2][R2]
        #         else:
        #             self.totalfrequency += 1
        # # for the ones above the criteria, we add it to the matrix

        # self.tsMatrix = self.FL
        # for E2 in list(self.FL[2]):
        #     for R2 in list(self.FL[2][E2]):
        #         if (E2, R2) not in self.tsMatrix:
        #             self.tsMatrix[(E2, R2)] = {}
                
        #         for S in self.database.sequences:
        #             self.tsMatrix[(E2, R2)][S.id] = 0
        #             if S.id in self.FL[2][E2][R2]:
        #                 self.tsMatrix[(E2, R2)][S.id] = len(self.FL[2][E2][R2][S.id])

                # for Si in list(self.FL[2][E2][R2]):
                    # only keeps the size so we can make feature space
                    # self.tsMatrix[(E2, R2)][Si] = len(self.FL[2][E2][R2][Si])

            #if len(self.FL[2][E2]) == 0:
            # del self.FL[2][E2]

    def chooseTopKFeatures(self, k, op = "vertical"):
                # if op == "vertical":
        #     countMatrix = {k: len(self.tsMatrix[k]) for k in self.tsMatrix}
        # elif op == "horizontal":
        #     countMatrix = {k: sum(self.tsMatrix[k].values()) for k in self.tsMatrix}
        # elif op == "disprop":
            
        #     ratios = []
        #     for lab in np.unique(self.labels):
        #         indices = self.labels == lab
        #         # set of instances of specific class
        #         set1 = data[indices]
        #         # set of the other instances
        #         set2 = data[~indices]
        #         # number of instances having specific feature inside the clase
        #         count1 = np.count_nonzero(set1)
        #         # number of instances having specific feature outside the class
        #         count2 = np.count_nonzero(set2)
        #         # calculate the risk ratio
        #         riskratio = (count1 / count2) / (len(set1) / len(set2))
        #         ratios.append(riskratio)
        #     ratios = np.array(ratios)
        #     if op2 == "max":
        #         summation = np.max(ratios, axis=0)
        #     if op2 == "average":
        #         summation = np.average(np.abs(ratios - 1), axis=0)
        # else:
        #     countMatrix = self.tsMatrix
        
        # 1. vertical or horizontal
        data_pd = pd.DataFrame.from_dict(self.FL).fillna(0)
        data = data_pd.to_numpy()
        summation = np.zeros(0)
        if k <= data.shape[1]:
            if op == "vertical":
                summation = np.count_nonzero(data, axis=0)
            elif op == "horizontal":
                # including vertical + horizontal
                summation = np.sum(data, axis=0)
            # Disproportionality or not?
            elif op == "relat_sup":
                ratios = []
                for lab in np.unique(self.labels):
                    indices = self.labels == lab
                    # set of instances of specific class
                    set1 = data[indices]
                    # number of instances having specific feature inside the clase
                    count1 = np.count_nonzero(set1, axis=0)
                    # calculate the risk ratio
                    riskratio = (count1 / len(set1))
                    ratios.append(riskratio)
                ratios = np.array(ratios)
                summation = np.average(ratios, axis=0)
            elif op in ["disprop_max", "disprop_avg"]:
                ratios = []
                for lab in np.unique(self.labels):
                    indices = self.labels == lab
                    # set of instances of specific class
                    set1 = data[indices]
                    # set of the other instances
                    set2 = data[~indices]
                    # number of instances having specific feature inside the clase
                    count1 = np.count_nonzero(set1, axis=0)
                    # number of instances having specific feature outside the class
                    count2 = np.count_nonzero(set2, axis=0)
                    # calculate the risk ratio
                    riskratio = ((count1+1) / (count2+1)) / (len(set1) / len(set2))
                    ratios.append(riskratio)
                ratios = np.array(ratios)

                if op == "disprop_max":
                    summation = np.max(ratios, axis=0)
                if op == "disprop_avg":
                    summation = np.average(np.abs(ratios - 1), axis=0)
            
            
            indices = np.argpartition(summation, -k)[-k:]
            selectedColumns = data_pd.iloc[:, indices].columns.values
            #keys = heapq.nlargest(k, countMatrix, key=countMatrix.get)
            # replace FL
            self.FL = {k: self.FL[k] for k in selectedColumns}

    def resetParams(self):
        # self.tempFL = defaultdict(lambda: defaultdict(set))
        self.FL = defaultdict(lambda: defaultdict(int))
        self.comparisoncount = 0
        self.totalfrequency = 0
        self.database = None
        
    def test(self, database):
        self.isTest = True
        self.resetParams()
        self.database = database
        for key in self.trainArrangementsSet:
            self.FL[key] = defaultdict(int)
        self.trainArrangementsSet = set(self.trainArrangementsSet)
        # self.pruneTestEventLabels()
        self.run()
        self.database = None
        return self.FL

    def train(self, database, labels):
        self.database = database
        self.labels = labels
        # if self.constraints["minSup"] != 0:
        # self.pruneWithSupport()
        self.run()
        # if constraints["k"] != 0:
        #     self.chooseTopKFeatures(constraints["k"], constraints["op"])
        self.trainArrangementsSet = list(self.FL.keys())
        # print(self.trainArrangementsSet)
        return self.FL
    
    def fit(self, X, y):
        # print(X.shape)
        X_train_split = np.array_split(X, self.split_no, axis=1)
        self.trainset_created = []
        for split_part in X_train_split:
            train_dataset = createDatabase(split_part, 
                            self.window_size, 
                            self.window_size_slope,
                            self.alphabet_size,
                            self.alphabet_size_slope,
                            self.glob)
            tsMatrix = self.train(train_dataset, y)
        self.vectorizer = DictVectorizer(dtype=np.uint16, sparse=True)
        # row_labels = list(trainset) 
        row_labels = list(tsMatrix) 
        #matrix = vectorizer.fit_transform([data[i] for i in row_labels]) 
        trainset = self.vectorizer.fit_transform([tsMatrix[i] for i in row_labels]).T
        # print(trainset.shape)
        self.trainset_created.append(trainset)
        trainset = hstack(self.trainset_created)
        # default classifier
        self.internal_classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
        self.internal_classifier.fit(trainset, y)
        return True

    def create_test_embedding(self, X, y):
        X_test_split = np.array_split(X, self.split_no, axis=1)
        self.testset_created = []
        for split_part in X_test_split:
            test_dataset = createDatabase(split_part, 
                            self.window_size, 
                            self.window_size_slope,
                            self.alphabet_size,
                            self.alphabet_size_slope,
                            self.glob)
        tsMatrix_test = self.test(test_dataset)
        row_labels = list(tsMatrix_test) 
        testset = self.vectorizer.fit_transform([tsMatrix_test[i] for i in row_labels]).T
        self.testset_created.append(testset)
        testset = hstack(self.testset_created) 
        return testset
    
    def predict(self, X, y):
        testset = self.create_test_embedding(X, y)
        labels = self.internal_classifier.predict(testset, y)
        return labels
    
    def score(self, X, y):
        testset = self.create_test_embedding(X, y)
        score = self.internal_classifier.score(testset, y)
        return score


    def run(self):
        self.createZTable()


        # Repeat this thing for x time
        #   1) choose one interval
        #   2) choose another intervals having that range
        #   3) choose n adjacent intervals and get frequent ones
        #    - maybe we can achieve that by modifying some gap constraint related thing
        #    - when we try to include the intervals, we have a gap, and we finish including process when we reach any interval exceeding that gap
        #      even before we get the pairwise relation -> even for some time series pairwise relation takes a lot of time
        #   4) if there is any frequent ones, increase the size
        #   5) horizontal support as a feature? ? -> like Z-Embedding

        
        # self.FL[3] = {}
        # # # # run the iteration growZ to grow and check frequent arrangements iteratively
        # for pair in self.FL[2]:
        #      for Rpair in self.FL[2][pair]:
        #          self.growZ(pair, Rpair, 3)
        
        #print("3. TOTAL COMPARISON COUNTS:", self.comparisoncount)
        #print("4. TOTAL FREQUENT ARRANGEMENTS:", self.totalfrequency)