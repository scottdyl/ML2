# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 19:16:22 2022

@author: drsco
"""
#%%% background
#This is called DeathToGridSearch because with this example you will never 
#have to think about how to manage a large number of classifiers etc 
#simultaneously.  You will now be able to run and collect results in a 
#very straightforward manner.  #LongLongLiveGridSearch!
#%%% imports
# Homework 2
import numpy as np
from sklearn.metrics import accuracy_score # other metrics too pls!
from sklearn.ensemble import RandomForestClassifier # more!
from sklearn.model_selection import KFold

import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
#%%% instructions
# adapt this code below to run your analysis
# 1. Write a function to take a list or dictionary of clfs and hypers(i.e. use logistic regression), each with 3 different sets of hyper parameters for each
# 2. Expand to include larger number of classifiers and hyperparameter settings
# 3. Find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function

#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html
#C:/Users/drsco/Documents/GitHub/StudentsPerformance.csv
#h/jbcroom/

#%%%
model_dict = {
    'RandomForestClassifier': RandomForestClassifier,
    'KNeighboursClassifier': KNeighborsClassifier,
    'LogisticRegression': LogisticRegression
}
#functions we want to use and their hyper parameters  
model_params_dict = {
    'RandomForestClassifier':{ 
            "n_estimators"      : [100, 200, 500, 1000],
            "max_features"      : ["auto", "sqrt", "log2"],
            "bootstrap": [True],
            "criterion": ['gini', 'entropy'],
            "oob_score": [True, False]
            },
    'KNeighboursClassifier': {
        'n_neighbors': np.arange(3,5,7,9,10,20),
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute']
        },
    'LogisticRegression': {
        'solver': ['newton-cg', 'sag', 'lbfgs'],
        'multi_class': ['ovr', 'multinomial']
        }  
}
#%%% given code by prof
M = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])
L = np.ones(M.shape[0])
n_folds = 5

data = (M, L, n_folds)

def run(a_clf, data, clf_hyper={}):
  M, L, n_folds = data # unpack data container
  kf = KFold(n_splits=n_folds) # Establish the cross validation
  ret = {} # classic explication of results

  for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
    clf = a_clf(**clf_hyper) # unpack parameters into clf is they exist
    clf.fit(M[train_index], L[train_index])
    pred = clf.predict(M[test_index])
    ret[ids]= {'clf': clf,
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred)}
  return ret

results = run(RandomForestClassifier, data, clf_hyper={})
#LongLongLiveGridS#LongLon#LLongLiveGridSearch!gLiveGridSearch!
#%%%

