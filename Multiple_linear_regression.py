#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 01:52:42 2020

@author: alienmoore
"""

#Multiple linear Regression

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values #matrix of feature(independant variable)
Y = dataset.iloc[:, 4].values #vectore of the dependent variable or target value

# Encoding Categorical Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')# the categorical variable is the 3rd index that's why we are using the index of 3
X = np.array(ct.fit_transform(X), dtype=np.float)

#Avoiding the Dummy variable Trap
X = X[:, 1:]

#splitting the dataset into the training set and Tes87t set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) # split the dataset into a training and test set

#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()#object of the class linear regression
regressor.fit(X_train, Y_train)#Fit the regressor to the training set

# Predicting the Test set result
y_pred = regressor.predict(X_test)# we will apply the regressor to predict the test set

#Building the optimal model using Backward elimination
import statsmodels.formula.api as sm
import statsmodels.regression.linear_model as lm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X,axis = 1)#np.ones will add 1 in the array by specifiying in the parentheses(row, col)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = lm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = lm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = lm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = lm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = lm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)