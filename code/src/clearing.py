import math
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.discriminant_analysis import StandardScaler
import seaborn as sns

data_matrix_corr="./src/static/data_matric_corr.png"
data_summary="./src/static/data_summury.png"

def data_clearing(data):

    """ Preprocess of the data after reading the csv file. It changes the name of some variables and 
        supress some variables based on correlation matrix """
    
    
    global challenge
    challenge =pd.read_csv(data,sep=';')
    rename_columns(challenge)
    delete_columns(challenge)
    check_numbers_not_null(challenge) 
    delete_columns_correlation(challenge)
    return challenge





def visualize(challenge):

    """Save figures that helps understanding the data """

    plt.figure(figsize=(10,10))
    sns.heatmap(challenge.corr(), annot=True, cmap="summer")
    plt.savefig(data_matrix_corr)
    plt.show()
  


    return  

    

def delete_columns_correlation(challenge):
     """ to delete columns that are no longer useful"""


     columns_null=[ 'nb_hour_total_coupure',
                   'nb_call_est',
                    "echec",
                    'nb_hour_echec',
                    'nb_hour_reiter_call',
                    'nb_hour_paging_without_response',
                    'nb_hour_call_est_bad_setupduration'
                    ]
     challenge.drop(columns=columns_null,inplace=True)


     return


def rename_columns(challenge):
    """to rename all the columns of the dataframe needed"""

    cahllenge=challenge.rename(columns={"qoe_challenge_table_for_model_v2.msisdn":"msisdn"},inplace=True)
    challenge.rename(columns={"qoe_challenge_table_for_model_v2.coupure_appel_nb_daily":"coupure"},inplace=True)
    challenge.rename(columns={"qoe_challenge_table_for_model_v2.nb_hour_coupure_appel_daily":"nb_hour_total_coupure"},inplace=True)
    challenge.rename(columns={"qoe_challenge_table_for_model_v2.echec_appel_nb_daily":"echec"},inplace=True)
    challenge.rename(columns={"qoe_challenge_table_for_model_v2.paging_without_response_nb_daily":"paging_without_response_nb"},inplace=True)
    challenge.rename(columns={"qoe_challenge_table_for_model_v2.nb_hour_paging_without_response_daily":"nb_hour_paging_without_response"},inplace=True)
    challenge.rename(columns={"qoe_challenge_table_for_model_v2.nb_hour_echec_appel_daily":"nb_hour_echec"},inplace=True)

    challenge.rename(columns={"qoe_challenge_table_for_model_v2.nb_call_est_daily":"nb_call_est"},inplace=True)
    challenge.rename(columns={"qoe_challenge_table_for_model_v2.reiter_call_interval_nb_daily":"reiter_call_interval_nb"},inplace=True)
    challenge.rename(columns={"qoe_challenge_table_for_model_v2.nb_hour_reiter_call_daily":"nb_hour_reiter_call"},inplace=True)
    challenge.rename(columns={"qoe_challenge_table_for_model_v2.call_est_bad_setupduration_nb_daily":"call_est_bad_setupduration_nb"},inplace=True)
    challenge.rename(columns={"qoe_challenge_table_for_model_v2.nb_hour_call_est_bad_setupduration_daily":"nb_hour_call_est_bad_setupduration"},inplace=True)

    return

def delete_columns(challenge):  
     """ to delete columns that are no longer useful"""



     columns_null=[
                    "qoe_challenge_table_for_model_v2.year",
                    "qoe_challenge_table_for_model_v2.imsi",
                    "qoe_challenge_table_for_model_v2.month",
                    "msisdn_H",
                  ]
     challenge.drop(columns=columns_null,inplace=True)


     return




def check_numbers_not_null(challenge):

    """Only keep elements with a test and msistn number not equal to 0 """
    challenge=challenge.loc[challenge["msisdn"]!=0]
    challenge=challenge.loc[challenge["test"]!=0]
    return challenge
