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
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import KFold
import itertools
import matplotlib.pyplot as plt
from sklearn import datasets
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
#https://scikit-learn.org/stable/datasets/real_world.html
#https://realpython.com/python-csv/
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html


#%%%
models = {
    'LogisticRegression': LogisticRegression,
    'KNeighboursClassifier': KNeighborsClassifier,
    #I have never used XGB but someone mentioned it and here we are
    #so far it is a pain in my ass
    'XGBClassifier': XGBClassifier,
    'RandomForestClassifier': RandomForestClassifier
}
#functions we want to use and their hyper parameters  
#I am setting a random state for consistancy
hyper_parms = {
    'LogisticRegression': {
        #change up the c values because they were too small
        'C': [100,250,500],
        'class_weight': ['balanced'],
#        'random_state': [42],
#i have to up the max iter to avid error since the data isn't fit for logisticReg
        'max_iter':[10000],
        'solver': ['newton-cg', 'sag', 'lbfgs'],
        'multi_class': ['ovr', 'multinomial']
        },    
    'KNeighboursClassifier': {
        'n_neighbors': [3,5,7,9,10,20],
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute'],
#        'random_state': [42]
          },
      'XGBClassifier': {
          'max_depth': range (2, 10, 1),
          'n_estimators': range(60, 220, 40),
          'learning_rate': [0.1, 0.01, 0.05],
          'use_label_encoder': [False],
          'eval_metric': ['mlogloss'],
#        'random_state': [42]
          },
      'RandomForestClassifier':{ 
          "n_estimators" : [100, 200, 500, 1000],
          'max_depth': [ 50, 100],
          "max_features" : ["auto", "sqrt", "log2"],
          "criterion": ['gini', 'entropy'],
          "oob_score": [True, False],
#        'random_state': [42]
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
#%%% data
#load in data
#good ol' iris from scikit-learn
n_folds = 5
iris_data, iris_target = datasets.load_iris(return_X_y=True)
data = (iris_data, iris_target,n_folds)
#%%% breakdown node
#help came from office hours to help unpack clfs for plotting
def groupClassifiers(results_dict):
    clfsAccuracyDict = {}
    
    for key in results_dict:
        k1 = results_dict[key]['clf']
        v1 = results_dict[key]['accuracy']
        k1Test = str(k1) 
        if k1Test in clfsAccuracyDict:
            clfsAccuracyDict[k1Test].append(v1) 
        else:
            clfsAccuracyDict[k1Test] = [v1] 
    return(clfsAccuracyDict)

#%%% plotting
def plot_parameters(clfsAccuracyDict,filename='clf_Histograms_'):
    filename_prefix = filename
    plot_num = 0
    n = max(len(v1) for k1, v1 in clfsAccuracyDict.items())

    #create the histograms
    #matplotlib is used to create the histograms: https://matplotlib.org/index.html
    for k1, v1 in clfsAccuracyDict.items():
        fig = plt.figure(figsize =(10,10))
        ax  = fig.add_subplot(1, 1, 1)
        plt.hist(v1, alpha=0.75)
        ax.set_title(k1)
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Frequency', fontsize=25)
        ax.yaxis.set_ticks(np.arange(0, n+1, 1)) 
        ax.grid(True) 
        plot_num_str = str(plot_num) 
        filename = filename_prefix + plot_num_str
        plt.savefig(filename, bbox_inches = 'tight')
        plot_num = plot_num+1 
    plt.show()
    
    for k1, v1 in clfsAccuracyDict.items():
        
        
        plt.show()

#%%% final model
def final_model(models, hyper_parms, data, filename=''):
    #empty dicts to add to later
    np_results = {}
    accuracyDics = {}
    #print out which model we are looking at
    for key, value in models.items():
        print('Processing Model: ', key)
        #this takes the specific key value for each model as we are looping
        hp_dict = hyper_parms[key]
        #this will zip together the key and all the possible values given in the hyper_parm
        keys1, values1 = zip(*hp_dict.items())
        #this creates a dictionary of each of the possible hyper_parms
        paramList = [dict(zip(keys1, v)) for v in itertools.product(*values1)]
        for dic in paramList: #this takes the dictionary created above and loops through it and runs them all
            np_results.update(run(value, data, dic))

        accuracyDics.update(groupClassifiers(np_results))


    #plot the parameters
    plot_parameters(accuracyDics,filename)
#%% actual gridsearch
from sklearn.model_selection import GridSearchCV

def fit_model(train_features, train_actuals):
        for name in models.keys():
            est = models[name]
            est_params = hyper_parms[name]
            gscv = GridSearchCV(estimator=est, param_grid=est_params, cv=5)
            gscv.fit(train_features, train_actuals)
            print("best parameters are: {}".format(gscv.best_estimator_))
#%%% Run
#my model
final_model(models, hyper_parms, data, "dylan_test_")
#"real" gridsearch
#fit_model(iris_target,iris_data)