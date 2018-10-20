# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 18:28:20 2018

# This program uses Machine Learning to
# determine the accuracy of predicting future trends based on historical
# prices with SVM model

@author: Leonardo Savasta
"""

# *** IMPORT LIBRARIES ***

import numpy as np
import pandas as pd
from collections import Counter
from sklearn import svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

# *** PROCESSING ***

def main():

    file = '../FOREX_dfs/DEXUSEU.csv'  
    
    ticker = 'DEXUSEU'
    
    accuracy = do_ml(ticker, file, dataCor = 90)
    
    print('Accuracy: ', accuracy)



# *** FUNCTIONS DECLARATION ***

# This function grabs the data from a csv file and cleans it for its use

def process_data_for_labels(ticker, file, dataCor = 30, days=7, index_change=False, drop_column=False, index=None, drop=None):
    
    days = days
    
    df = pd.read_csv(file, index_col=0)
    
    dataLen = len(df.iloc[:,0])
    
    if(index_change):
        df.set_index(index, inplace=True)
    
    if(drop_column):
        df.drop(df.columns[drop], axis=1, inplace=True)
    
    
    df.reset_index(inplace=True)
    df = df.drop('DATE', axis=1)
    
    df = df.iloc[::-1]
    
    expData = pd.DataFrame()
    
    arraytemp = np.ndarray((dataCor,dataLen-dataCor))
    
    for x in range(0, dataCor):
        arraytemp[x] = df.iloc[x:dataLen-(dataCor-x),0]
        
    arraytemp = arraytemp.transpose()
    
    expData = pd.DataFrame(arraytemp)
    
    
    for i in range(1, days+1):
        df['{}_close'.format(i)] = (df.iloc[:,0].shift(-i) - df.iloc[:,0]) / df.iloc[:,0]
        
    df = df.drop(df.index[0:dataCor])
    
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    
    expData.fillna(0, inplace=True)
    expData = expData.replace([np.inf, -np.inf], np.nan)
    expData.dropna(inplace=True)
    
    return df, expData


# This function analyzes columns and returns:
#   1 - If the data changes positively by more than %2
#  -1 - If the data changes negatively by more than %2
#   0 - If the data does not change by more than %2 in any direction

def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.01
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0

# This function does data analysis by applying buy_sell_hold to every data point
# Returns: - A two dimension array that include only the values of the dataframe
#          - An array with the values from the target column in the dataframe.
#          - The dataframe itself

def extract_featuresets(ticker, file, dataCor=30, days=7, index_change=False, drop_column=False, index=None, drop=None):
    
    df, expData = process_data_for_labels(ticker, file, dataCor, days, index_change, drop_column, index, drop)

    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                               df['1_close'],
                                               df['2_close'],
                                               df['3_close'],
                                               df['4_close'],
                                               df['5_close'],
                                               df['6_close'],
                                               df['7_close'] ))
    
    targets = df['{}_target'.format(ticker)].values.tolist()
    str_targets = [str(i) for i in targets]
    print('Data spread:', Counter(str_targets))
        
    df_vals = expData
    
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)
    
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values
    
    return X, y, df

# This function performs machine learning on the dataframe. It perfomrs a
# cross validation to learn the correlations in 75% of the data and then tests
# the model in 25% of the data. The function returns the accuracy of the model
# according ot the performed test.

def do_ml(ticker, file, dataCor=30, days=7, index_change=False, drop_column=False, index=None, drop=None):

    X, y, df = extract_featuresets(ticker, file, dataCor, days, index_change, drop_column, index, drop)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
        
    clf = VotingClassifier([('linear', svm.LinearSVC()),
                            ('knc', neighbors.KNeighborsClassifier()),
                            ('random', RandomForestClassifier())])

    clf.fit(X_train, y_train)
    
    confidence = clf.score(X_test, y_test)
    
    predictions = clf.predict(X_test)
    print('Predicted class counts:', Counter(predictions))
    
    return confidence

    
# Run the program

main()

