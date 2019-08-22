#https://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
#https://www.scribd.com/doc/315101213/Python-for-Probability-Statistics-And-Machine-Learning
#https://www.scribd.com/read/365183554/Python-Real-World-Machine-Learning#



import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from itertools import product
from sklearn import datasets
import datetime
#from __future__ import print_function
from time import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print(__doc__)


import seaborn as sns 
#%matplotlib inline
import matplotlib.pyplot as plt 

from matplotlib.backends.backend_pdf import PdfPages

from reportlab.pdfgen import canvas
from reportlab.lib.units import inch, cm

# adapt this code below to run your analysis


# Due before live class 2
# 1. Write a function to take a list or dictionary of clfs and hypers ie use logistic regression, each with 3 different sets of hyper parrameters for each
#Due before live class 3
# 2. expand to include larger number of classifiers and hyperparmater settings
# 3. find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings
#Due before live class 4
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function

#M = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])
#L = np.ones(M.shape[0])
#M = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])
#L = np.random.choice([0,1], size=(M.shape[0],), p=[1./3, 2./3])
iris = datasets.load_iris()
#X-axis
M = iris.data[:, :2]
#Y-axis
L = iris.target

# Update to change the # of folds in K-fold CV
n_folds = 5

data = (M, L, n_folds)

#M, L, n_folds = data
#kf = KFold(n_splits=n_folds)
#Office Hours 1/12 and 1/13, 2019
def run(a_clf, data, clf_hyper={}):
  M, L, n_folds = data # unpack data containter
  kf = KFold(n_splits=n_folds) # Establish the cross validation
  ret = {} # classic explicaiton of results
  for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
      clf = a_clf(**clf_hyper)
      clf.fit(M[train_index], L[train_index])
      pred = clf.predict(M[test_index])
      ret[ids]= {'clf': clf,                    #EDIT: Create arrays of
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred)}    
  return ret

def runGridSearch(a_clf, data, clf_hyper={}):
  M, L, n_folds = data # unpack data containter
  kf = KFold(n_splits=n_folds) # Establish the cross validation
  ret = {} # classic explicaiton of results
  for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
      clf = a_clf(**clf_hyper)
      clf.GridSearchCV(M[train_index], L[train_index])
      pred = clf.predict(M[test_index])
      ret[ids]= {'clf': clf,                    #EDIT: Create arrays of
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred)}    
  return ret

#Takes results given from the run function and gets each classifier/hyper parameter combination, a set of results into a dictionary for visualization
def populateClfAccuracyDict(results):
    for key in results:
        k1 = results[key]['clf'] 
        v1 = results[key]['accuracy']
        k1Test = str(k1)
        
        k1Test = k1Test.replace('            ',' ') # remove large spaces from string
        k1Test = k1Test.replace('          ',' ')
        if k1Test in clfsAccuracyDict:
            clfsAccuracyDict[k1Test].append(v1)
        else:
            clfsAccuracyDict[k1Test] = [v1]

def myHyperSetSearch(clfsList,clfDict):
    for clf in clfsList:
        clfString = str(clf)
        
        for k1, v1 in clfDict.items():
            if k1 in clfString:
                k2,v2 = zip(*v1.items())
                for values in product(*v2):
                    hyperSet = dict(zip(k2, values))
                    results = run(clf, data, hyperSet)
                    populateClfAccuracyDict(results)
                    
            
  #for clfs in a_clf:
  #  for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
  #      clf = clfs(**clf_hyper)
  #      clf.fit(M[train_index], L[train_index])
  #      ret[ids]= {'clf': clf,                    #EDIT: Create arrays of
  #                   'train_index': train_index,
  #                   'test_index': test_index
                     #,'accuracy': accuracy_score(L[test_index], M[y_pred])
  #                   }
  #return ret 
clfsList = [RandomForestClassifier, LogisticRegression, SVC] 
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
#https://scikit-learn.org/stable/modules/svm.html

clfDict = {'RandomForestClassifier': {"min_samples_split": [2,3,4]}, 'LogisticRegression': {"tol": [0.01,0.1, .05], "solver": ['lbfgs', 'sag', 'saga']}, 'SVC': {"tol": [0.01,0.1, .05]}}
#multi_class: ['ovr', 'multinomial', 'auto'],
#clfsList = [LogisticRegression] 
clfsAccuracyDict = {}


myHyperSetSearch(clfsList,clfDict) 
#print(clfsAccuracyDict)

# for determining maximum frequency (# of kfolds) for histogram y-axis
n = max(len(v1) for k1, v1 in clfsAccuracyDict.items())

# for naming the plots
filename_prefix = 'clf_Histograms_'

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
               

#results = run(clfsList, data, clf_hyper={})

#print(results)

# Office Hours 1/19 and 1/20, 2019
#https://medium.com/@neuralnets/data-visualization-with-python-and-seaborn-part-1-29c9478a8700
with PdfPages('HW1_Death_to_GridSearch_Lingle.pdf') as pdf:
    for k1, v1 in clfsAccuracyDict.items():
# for each key in our clfsAccuracyDict, create a new histogram with a given key's values 
        fig = plt.figure(figsize =(20,10)) # This dictates the size of our histograms
        ax  = fig.add_subplot(1, 1, 1) # As the ax subplot numbers increase here, the plot gets smaller
        #ax = sns.boxplot
        plt.hist(v1, facecolor='green', alpha=0.75) # create the histogram with the values
        ax.set_title(k1, fontsize=30) # increase title fontsize for readability
        ax.set_xlabel('Classifer Accuracy (By K-Fold)', fontsize=20) # increase x-axis label fontsize for readability
        ax.set_ylabel('Frequency', fontsize=20) # increase y-axis label fontsize for readability
        ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1)) # The accuracy can only be from 0 to 1 (e.g. 0 or 100%)
        ax.yaxis.set_ticks(np.arange(0, n+1, 1)) # n represents the number of k-folds
        ax.xaxis.set_tick_params(labelsize=20) # increase x-axis tick fontsize for readability
        ax.yaxis.set_tick_params(labelsize=20) # increase y-axis tick fontsize for readability
        #ax.grid(True) # you can turn this on for a grid, but I think it looks messy here.

        # pass in subplot adjustments from above.
        plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
        plot_num_str = str(plot_num) #convert plot number to string
        #filename = filename_prefix + plot_num_str # concatenate the filename prefix and the plot_num_str
        #plt.savefig(filename, bbox_inches = 'tight') # save the plot to the user's working directory as .png by default
        plot_num = plot_num+1 # increment the plot_num counter by 1
        d = pdf.infodict()
        d['Title'] = 'HW 1: Death to Grid Search'
        d['Author'] = u'Jason Lingle'
        d['Subject'] = 'HW1: Death to Grid Search'
        #d['Keywords'] = 'PdfPages multipage keywords author title subject'
        #d['CreationDate'] = datetime.datetime(2009, 11, 13)
        d['ModDate'] = datetime.datetime.today()
        pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
        plt.close()
    
#plt.show()



#https://www.scribd.com/read/365183554/Python-Real-World-Machine-Learning#t_search-menu_829698

# Loading the Digits dataset
#digits = datasets.load_digits()
#
## To apply an classifier on this data, we need to flatten the image, to
## turn the data in a (samples, feature) matrix:
#n_samples = len(digits.images)
#X = digits.images.reshape((n_samples, -1))
#y = digits.target
#
## Split the dataset in two equal parts
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.5, random_state=0)
#
## Set the parameters by cross-validation
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                     'C': [1, 10, 100, 1000]},
#                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
#
#scores = ['precision', 'recall']
#
#for score in scores:
#    print("# Tuning hyper-parameters for %s" % score)
#    print()
#
#    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
#                       scoring='%s_macro' % score)
#    clf.fit(X_train, y_train)
#
#    print("Best parameters set found on development set:")
#    print()
#    print(clf.best_params_)
#    print()
#    print("Grid scores on development set:")
#    print()
#    means = clf.cv_results_['mean_test_score']
#    stds = clf.cv_results_['std_test_score']
#    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#        print("%0.3f (+/-%0.03f) for %r"
#              % (mean, std * 2, params))
#    print()
#
#    print("Detailed classification report:")
#    print()
#    print("The model is trained on the full development set.")
#    print("The scores are computed on the full evaluation set.")
#    print()
#    y_true, y_pred = y_test, clf.predict(X_test)
#    print(classification_report(y_true, y_pred))
#    print()
#
## Note the problem is too easy: the hyperparameter plateau is too flat and the
## output model is the same for precision and recall with ties in quality.
#n_samples = len(digits.images)
#X = digits.images.reshape((n_samples, -1))
#y = digits.target
#
## Split the dataset in two equal parts
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.5, random_state=0)
#
## Set the parameters by cross-validation
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                     'C': [1, 10, 100, 1000]},
#                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
#
#scores = ['precision', 'recall']
#
#for score in scores:
#    print("# Tuning hyper-parameters for %s" % score)
#    print()
#
#    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
#                       scoring='%s_macro' % score)
#    clf.fit(X_train, y_train)
#
#    print("Best parameters set found on development set:")
#    print()
#    print(clf.best_params_)
#    print()
#    print("Grid scores on development set:")
#    print()
#    means = clf.cv_results_['mean_test_score']
#    stds = clf.cv_results_['std_test_score']
#    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#        print("%0.3f (+/-%0.03f) for %r"
#              % (mean, std * 2, params))
#    print()
#
#    print("Detailed classification report:")
#    print()
#    print("The model is trained on the full development set.")
#    print("The scores are computed on the full evaluation set.")
#    print()
#    y_true, y_pred = y_test, clf.predict(X_test)
#    print(classification_report(y_true, y_pred))
#    print()


# starter code for GridSearchCV with multiple classifiers.
# https://stackoverflow.com/questions/50265993/alternate-different-models-in-pipeline-for-gridsearchcv


import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from itertools import product

iris = datasets.load_iris()
#X-axis
M = iris.data[:, :2]
#Y-axis
L = iris.target

# Update to change the # of folds in K-fold CV
n_folds = 5

data = (M, L, n_folds)

clfsAccuracyDict = {}

# the models that you want to compare
models = {
    'RandomForestClassifier': RandomForestClassifier(),
    'KNeighboursClassifier': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression()
}

# the optimisation parameters for each of the above models
param_grid = {
    'RandomForestClassifier':{ 
            "n_estimators"      : [100, 200, 500, 1000],
            "max_features"      : ["auto", "sqrt", "log2"],
            "bootstrap": [True],
            "criterion": ['gini', 'entropy'],
            "oob_score": [True, False]
            },
    'KNeighboursClassifier': {
        'n_neighbors': np.arange(3, 15),
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute']
        },
    'LogisticRegression': {
        'solver': ['newton-cg', 'sag', 'lbfgs'],
        'multi_class': ['ovr', 'multinomial']
        }  
}
    

def fit(train_features, train_actuals):
        """
        fits the list of models to the training data, thereby obtaining in each 
        case an evaluation score after GridSearchCV cross-validation
        """
        for name in models.keys():
            est = models[name]
            est_params = param_grid[name]
            gscv = GridSearchCV(estimator=est, param_grid=est_params, cv=5)
            gscv.fit(train_actuals, train_features)
            print("best parameters are: {}".format(gscv.best_estimator_))
            
results = fit

n_folds = 5
# Step 1: bring in your matrix and labels
def runGridSearch(a_clf, data, param_grid=param_grid):
  M, L, n_folds = data # unpack data containter
  kf = KFold(n_splits=n_folds) # Establish the cross validation
  ret = {} # classic explicaiton of results
  for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
      clf = a_clf(**param_grid)
      clf.GridSearchCV(M[train_index], L[train_index])
      pred = clf.GridSearchCV(M[test_index])
      #kf = KFold(n_splits=n_folds) # Establish the cross validation
      ret[ids]= {'clf': clf,                    #EDIT: Create arrays of
               'train_index': train_index,
               'test_index': test_index}
      return ret
               
               

# Step 2: set up train and test subsets
# Step 3: use the fit function above.
def populateClfAccuracyDict(results):
    for key in results:
        k1 = results[key]['clf'] 
        v1 = results[key]['accuracy']
        k1Test = str(k1)
        
        k1Test = k1Test.replace('            ',' ') # remove large spaces from string
        k1Test = k1Test.replace('          ',' ')
        if k1Test in clfsAccuracyDict:
            clfsAccuracyDict[k1Test].append(v1)
        else:
            clfsAccuracyDict[k1Test] = [v1]

def myGridSearch(clfsList,clfDict):
    for clf in clfsList:
        clfString = str(clf)
        
        for k1, v1 in clfDict.items():
            if k1 in clfString:
                k2,v2 = zip(*v1.items())
                for values in product(*v2):
                    hyperSet = dict(zip(k2, values))
                    results = runGridSearch(clf, data, hyperSet)
                    populateClfAccuracyDict(results)
                    