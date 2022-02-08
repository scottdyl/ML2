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
from xgboost import XGBClassifier
#%%% instructions
# adapt this code below to run your analysis
# 1. Write a function to take a list or dictionary of clfs and hypers(i.e. use logistic regression), each with 3 different sets of hyper parameters for each
# 2. Expand to include larger number of classifiers and hyperparameter settings
# 3. Find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function

#%%% resources
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html
#C:/Users/drsco/Documents/GitHub/StudentsPerformance.csv
#h/jbcroom/
#https://scikit-learn.org/stable/datasets/real_world.html
#https://realpython.com/python-csv/
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html


#%%%
model_dict = {
    'LogisticRegression': LogisticRegression,
    'KNeighboursClassifier': KNeighborsClassifier,
    #I have never used XGB but someone mentioned it and here we are
    #so far it is a pain in my ass
    'XGBClassifier': XGBClassifier,
    'RandomForestClassifier': RandomForestClassifier
}
#functions we want to use and their hyper parameters  
model_params_dict = {
    'LogisticRegression': {
        'solver': ['newton-cg', 'sag', 'lbfgs'],
        'multi_class': ['ovr', 'multinomial']
        },    
    'KNeighboursClassifier': {
        'n_neighbors': [3,5,7,9,10,20],
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute']
        },
    'XGBClassifier': {
        'max_depth': range (2, 10, 1),
        'n_estimators': range(60, 220, 40),
        'learning_rate': [0.1, 0.01, 0.05],
        'use_label_encoder': [False],
        'eval_metric': ['mlogloss']
        },
    'RandomForestClassifier':{ 
        "n_estimators"      : [100, 200, 500, 1000],
        "max_features"      : ["auto", "sqrt", "log2"],
        "bootstrap": [True],
        "criterion": ['gini', 'entropy'],
        "oob_score": [True, False]
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
#load in data
#good ol' iris from scikit-learn
iris_data, iris_target = datasets.load_iris(return_X_y=True)
data = (iris_data, iris_target,n_folds)
#%%%
def groupClassifiers(results_dict):
    clfsAccuracyDict = {}
    
    for key in results_dict:
        k1 = results_dict[key]['clf']
        v1 = results_dict[key]['accuracy']
        k1Test = str(k1) #Since we have a number of k-folds for each classifier...
                         #We want to prevent unique k1 values due to different "key" values
                         #when we actually have the same classifer and hyper parameter settings.
                         #So, we convert to a string

        #String formatting
        k1Test = k1Test.replace('            ',' ') # remove large spaces from string
        k1Test = k1Test.replace('          ',' ')

        #Then check if the string value 'k1Test' exists as a key in the dictionary
        if k1Test in clfsAccuracyDict:
            clfsAccuracyDict[k1Test].append(v1) #append the values to create an array (techically a list) of values
        else:
            clfsAccuracyDict[k1Test] = [v1] #create a new key (k1Test) in clfsAccuracyDict with a new value, (v1)

    return(clfsAccuracyDict)

#%%%
def plot_parameters(clfsAccuracyDict,filename='clf_Histograms_'):
    # for naming the plots
    filename_prefix = filename

    # initialize the plot_num counter for incrementing in the loop below
    plot_num = 1

    # Adjust matplotlib subplots for easy terminal window viewing
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.6      # the top of the subplots of the figure
    wspace = 0.2   # the amount of width reserved for space between subplots,
                   # expressed as a fraction of the average axis width
    hspace = 0.2   # the amount of height reserved for space between subplots,
                   # expressed as a fraction of the average axis height
                   
    # for determining maximum frequency (# of kfolds) for histogram y-axis
    n = max(len(v1) for k1, v1 in clfsAccuracyDict.items())

    #create the histograms
    #matplotlib is used to create the histograms: https://matplotlib.org/index.html
    for k1, v1 in clfsAccuracyDict.items():
        # for each key in our clfsAccuracyDict, create a new histogram with a given key's values
        fig = plt.figure(figsize =(10,10)) # This dictates the size of our histograms
        ax  = fig.add_subplot(1, 1, 1) # As the ax subplot numbers increase here, the plot gets smaller
        plt.hist(v1, facecolor='green', alpha=0.75) # create the histogram with the values
        ax.set_title(k1, fontsize=25) # increase title fontsize for readability
        ax.set_xlabel('Classifer Accuracy (By K-Fold)', fontsize=25) # increase x-axis label fontsize for readability
        ax.set_ylabel('Frequency', fontsize=25) # increase y-axis label fontsize for readability
        ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1)) # The accuracy can only be from 0 to 1 (e.g. 0 or 100%)
        ax.yaxis.set_ticks(np.arange(0, n+1, 1)) # n represents the number of k-folds
        ax.xaxis.set_tick_params(labelsize=20) # increase x-axis tick fontsize for readability
        ax.yaxis.set_tick_params(labelsize=20) # increase y-axis tick fontsize for readability
        #ax.grid(True) # you can turn this on for a grid, but I think it looks messy here.

        # pass in subplot adjustments from above.
        plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
        plot_num_str = str(plot_num) #convert plot number to string
        filename = filename_prefix + plot_num_str # concatenate the filename prefix and the plot_num_str
        plt.savefig(filename, bbox_inches = 'tight') # save the plot to the user's working directory
        plot_num = plot_num+1 # increment the plot_num counter by 1
    plt.show()

#%%%
def hyper_search(model_dict, param_dict, data, filename=''):
    # define empty dictionaries to start
    np_results = {}
    accuracyDics = {}
    
    # iterate through the model dictionary to execute each model
    for key, value in model_dict.items():
        # just for grins, let the user know which model we're processing
        print('Processing Model: ', key)
        
        # get the hyper parameter dictionary listings for the specific model
        paramDict = param_dict[key]
        
        # take our hyper parameter dictionary and use itertools to build out
        # all possible permutations for execution
        keys1, values1 = zip(*paramDict.items())
        paramList = [dict(zip(keys1, v)) for v in itertools.product(*values1)]
        
        # iterate through the hyper parameter permutations and execute them
        for dic in paramList:
            # execute the run function on each model type and hyper parameter configuration
            # add the results to the np_results dictionary for use in other methods
            np_results.update(run(value, data, dic))
           # results = run(value, data, dic)
            
        # find the classifiers for plotting from all the permutations we've run through
        # this will get us to the "best" permutation of results to plot and prevent us
        # from printing 100's of plots
        accuracyDics.update(groupClassifiers(np_results))


    #plot the parameters
    plot_parameters(accuracyDics,filename)
#%%% Run
hyper_search(model_dict, model_params_dict, data, "dylan_test_")