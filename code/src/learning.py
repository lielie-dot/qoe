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

matrix_corr_cible="./src/static/matrix_cor_cible.png"
segmentation_results="./src/static/matrix_kmeans.png"



def unsupervised_supervised_training_given_test(data, file):


    """It aims at testing the model with measured data corresponding to a qoE (just the class for now) 
    already mesuread.
    It gives the accuracy score and th recall score of the model"""
    if file==" ":
        return "Write a correct name for the file"
    X_test_user=pd.read_csv(file,sep=';')
    if len(X_test_user.columns.to_list())!=5:
        return None
    else:

        return unsupervised_supervised_training_given(data,X_test_user,0)



def unsupervised_supervised_training_given_prediction(data,column1,column2,column3,column4):

    """   It aims at giving the prediction of the qoE giving informations written by the user  """


    if len(column1)==0 or len(column2)==0 or len(column3)==0 or len(column4)==0:
        print("Please write correctly the values")
        return -1,-1
    
    if not isinstance(column1[0], int) or not isinstance(column2[0], int) or not isinstance(column3[0], int)or not isinstance(column4[0], int):
        print("Please only enter integer values")
        return -1,-1
    
    columns={
             'coupure':column1,
             'paging_without_response_nb':column2,
             'reiter_call_interval_nb':column3,
            'call_est_bad_setupduration_nb':column4}
    X_test_user = pd.DataFrame(columns)

    return unsupervised_supervised_training_given(data,X_test_user,1)




def unsupervised_supervised_training_given(data,X_test_user,case):

    """ We first do a segmentation and then select
        the clusters that are considered as good and the one as not good.
        After use the logistic model regression to deduct the score and if the QoeE is good or not good """
    
    global data_cleared
    global data_cleared_after_kmeans
    global base_model_2


    data_cleared=data_clearing(data)
    base_model_2=segmentation(data_cleared)
    
    base_model_2['cible']=base_model_2["Cluster"]
    base_model_2['cible']=base_model_2['cible'].apply(find_cible)


    #train_tree(base_model_2,X_test_user,case)
    #train_mlp(base_model_2,X_test_user,case)
    return train_regression_logistic(base_model_2,X_test_user,case)
   
    


def find_cible(x):

    """this function aims at creating a label 1 for the good clusters and 0 for the bad clusters"""
    if x==0 or x==3:
        return 1
    else:
        return 0
    



def visualize_training_steps(data):

    """If we want to see the different steps of the clustering and clearing of the data"""

    data_cleared=data_clearing(data)
    base_model_2=segmentation(data_cleared)

    columnList1=base_model_2.columns.to_list()
    columnList1.remove("test")
    columnList1.remove("msisdn")
    base_model_2=base_model_2[columnList1]

    plt.figure(figsize=(15,6))
    sns.heatmap(base_model_2.groupby("Cluster").mean(), annot=True, cmap="summer")
    plt.savefig(segmentation_results)

    base_model_2['cible']=base_model_2["Cluster"]
    base_model_2['cible']=base_model_2['cible'].apply(find_cible)

    plt.figure(figsize=(10,10))
    sns.heatmap(base_model_2.corr(), annot=True, cmap="summer")
    plt.savefig(matrix_corr_cible)
    
    return
  




def train_regression_logistic(base_model_2,X_test_user,case):
    
    """ Perform a logistic regression and then show the score and the result of the prediction"""
    """According to the case given, it might also test the model with a set of data entered by the user"""
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


    scalar.fit(X_test_user)
    scalar.transform(X_test_user)

    
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2)
    ros = RandomOverSampler (random_state= 42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    print(f"number of 1 target : {y_train_resampled['cible'].value_counts()[1]}\n")
    print(f"number of 0 target : {y_train_resampled['cible'].value_counts()[0]}\n")

    print("\nTraining:\n")
    clf = LogisticRegression(C=100,random_state=24)
    clf.fit(X_train_resampled, y_train_resampled)

    coef = clf.coef_[0]
    intercept = clf.intercept_
    y_pred_train=y_pred=clf.predict(X_train)
    print(f"recall_score  for training data : {recall_score(y_train, y_pred_train)}")


    print("\nPrediction on validationn data:\n")
    y_pred=clf.predict(X_test)
    print(f"accuracy_score of the model on validation data : {accuracy_score(y_test,y_pred)}")
    print(f"recall_score  for validation data : {recall_score(y_test, y_pred)}")
    print(f"f1_score  for validation data : {f1_score(y_test, y_pred)}")
  


    if case==0:
     
        y_test_user=X_test_user["qoe"].to_list()
    
        X_test_user=X_test_user.drop("qoe",axis=1)
        y_pred_user=[]
        for k in range(0,len(y_test_user)):
            y_pred_user.append(y_value_sigmoid(X_test_user.loc[k,:].to_list(),coef,intercept))
        

        print(f"mean sqared error test data : {mean_squared_error(y_test_user,y_pred_user)}")
        return mean_squared_error(y_test_user,y_pred_user)
    if case==1:
        print("\nPrediction of QoE based on input data :\n")
        y_pred_user=clf.predict(X_test_user)
        
        print(f"Class predicted : {y_pred_user[0]}")
        print(f"QoE Score predicted : {y_value_sigmoid(X_test_user.iloc[0,:].to_list(),coef,intercept)}")
        return y_pred_user[0],y_value_sigmoid(X_test_user.iloc[0,:].to_list(),coef,intercept)

   
def y_value_sigmoid(x,coef,intercept):

    """Helps at calculating the score of QoE"""
    value=0

    for k in range(0,len(coef)):
        value=value+x[k]*coef[k] 
    value+=intercept[0]

    return round(100 / (1 + math.exp(-value)),2)



    

def train_tree(base_model_2,column1,column2,column3,column4):
    """Another model evaluated, for the prediction of the Qoe, using random forest"""


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
    clf = RandomForestClassifier(n_estimators=300,criterion='gini')
    clf.fit(X_train_resampled, y_train_resampled)
    print(f"score of the model : {clf.score(X_train_resampled,y_train_resampled)}")

    print("\nPrediction on test data:\n")
    y_pred=clf.predict(X_test)
    print(f"accuracy_score of the model on test data : {accuracy_score(y_test,y_pred)}")
    print(f"recall_score  for test data : {recall_score(y_test, y_pred)}")
    print(f"f1_score  for test data : {f1_score(y_test, y_pred)}")
  

    return



def train_mlp(base_model_2,column1,column2,column3,column4):
    """Another model evaluated, for the prediction of the Qoe, using neural networks"""


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
    clf = MLPClassifier()
    clf.fit(X_train_resampled, y_train_resampled)
    print(f"score of the model : {clf.score(X_train_resampled,y_train_resampled)}")


    print("\nPrediction on test data:\n")
    y_pred=clf.predict(X_test)
    print(f"accuracy_score of the model on test data : {accuracy_score(y_test,y_pred)}")
    print(f"recall_score  for test data : {recall_score(y_test, y_pred)}")
    print(f"f1_score  for test data : {f1_score(y_test, y_pred)}")
  

    return