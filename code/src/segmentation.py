
import math
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.discriminant_analysis import StandardScaler
import seaborn as sns

segmentation_results="./src/static/matrix_kmeans.png"

def segmentation(data_cleared):

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


    """ print("\nLooking for best params:\n")
    param_grid={'max_iter':[100,200,300,300],'random_state':[24,42,100,200,300,1000]}
    grid_search=GridSearchCV(clf,param_grid,cv=5)
    grid_search.fit(X)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Best Parameters:", best_params)
    print("Best Score:", best_score)"""


    data_cleared_bis=data_cleared[columnList1]

     
  
    data_cleared_bis["Cluster"] = clf.predict(X)

    data_cleared["Cluster"]=clf.predict(X)


    return data_cleared