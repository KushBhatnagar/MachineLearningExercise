# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 08:12:17 2019

@author: Kush
Problem Statement: A digital media company (similar to Voot, Hotstar, Netflix, etc.) had launched a show. Initially, the show got a good response, but then witnessed a decline in viewership.
The company wants to figure out what went wrong.
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('mediacompany.csv')
dataset = dataset.drop('Unnamed: 7',axis = 1)
dataset = dataset.drop('Date',axis = 1)
X=dataset.iloc[:,1:].values
y=dataset.iloc[:,0].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test set results
y_pred=regressor.predict(X_test)

#Graph for Predicted Vs Actual
x_axis = np.linspace(100000,750000,16)
fig = plt.figure()
plt.plot(x_axis,y_test, color="red",  linewidth=2.5, linestyle="-")
plt.plot(x_axis,y_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Actual Vs Predicted ', fontsize=20)

#Building the optimal model using Backward Elimination 
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((80,1)).astype(int),axis = 1,values=X)

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#X_opt = X[:, [0,3]]
#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#regressor_OLS.summary()

X_train_new, X_test_new, y_train, y_test = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)
regressor_new=LinearRegression()
regressor_new.fit(X_train_new,y_train)
y_pred_new=regressor_new.predict(X_test_new)

#Graph for New Predicted Vs Actual
fig = plt.figure()
plt.plot(x_axis,y_test, color="red",  linewidth=2.5, linestyle="-")
plt.plot(x_axis,y_pred_new, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Actual Vs _New Predicted ', fontsize=20)

#Graph for Predicted Vs New Predicted Vs Actual
fig = plt.figure()
plt.plot(x_axis,y_test, color="red",  linewidth=2.5, linestyle="-")
plt.plot(x_axis,y_pred_new, color="green", linewidth=2.5, linestyle="-")
plt.plot(x_axis,y_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Actual Vs Predicted Vs New Predicted ', fontsize=20)

#As if now this approach is not giving correct prediction, need to work on approach