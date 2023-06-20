


import math
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, mean_squared_error, recall_score,accuracy_score
from sklearn.discriminant_analysis import StandardScaler
import seaborn as sns
from clearing import *
from segmentation import *


segmentation_results="./src/static/matrix_kmeans.png"

def segmentation_parameters(data_cleared):

    """Based on the work previously done to select the right number of clusters (see formation_ngal folder) 
        we  do the segmentation with kmeans algorithm. After we visualize the result to indentify clusters proporties
        according to the goeal of this project  """
    
    columnList1=data_cleared.columns.to_list()
    columnList1.remove("test")
    columnList1.remove("msisdn")

    X = data_cleared.loc[:,columnList1]
    scalar=StandardScaler()
    scalar.fit(X)
    scalar.transform(X)
    clf = KMeans(n_clusters=4,max_iter=100,random_state=42)
    clf.fit(X)


    print("\nLooking for best params:\n")
    param_grid={'max_iter':[100,200,300,300],'random_state':[24,42,100,200,300,1000]}
    grid_search=GridSearchCV(clf,param_grid,cv=5)
    grid_search.fit(X)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Best Parameters:", best_params)
    print("Best Score:", best_score)

    return 







def train_regression_logistic_parameters(base_model_2,column1,column2,column3,column4):
    
    """ Perform a logistic regression and then show the score and the result of the prediction"""
    columnList = base_model_2.columns.to_list()
    columnList.remove("Cluster")
    columnList.remove("test")
    columnList.remove("msisdn")
    columnList.remove("cible")
   
    X = pd.DataFrame(base_model_2.loc[:, columnList])
    y = pd.DataFrame(base_model_2.loc[:, ["cible"]])
    scalar=MinMaxScaler()
    scalar.fit(X)
    scalar.transform(X)


    columns={
             'coupure':column1,
             'paging_without_response_nb':column2,
             'reiter_call_interval_nb':column3,
            'call_est_bad_setupduration_nb':column4}
    X_test_user = pd.DataFrame(columns)
    scalar.fit(X_test_user)
    scalar.transform(X_test_user)

    
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2)
    ros = RandomOverSampler (random_state= 42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
   
    print("\nTraining:\n")
    clf = LogisticRegression(C=100,random_state=24)
    clf.fit(X_train_resampled, y_train_resampled)
    print("\nLooking for best params:\n")
    param_grid={'C':[100,200,300,400,500,600,1000],'random_state':[24,42,100,200,300,1000]}
    grid_search=GridSearchCV(clf,param_grid,cv=5)
    grid_search.fit(X_train_resampled,y_train_resampled)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Best Parameters:", best_params)
    print("Best Score:", best_score)


    return









