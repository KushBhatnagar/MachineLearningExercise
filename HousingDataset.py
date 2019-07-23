# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 07:39:51 2019

@author: Kush
Implementing 'Housing' data set with another course approach
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Housing.csv')
X=dataset.iloc[:,1:].values
y=dataset.iloc[:,0].values

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,4] = labelencoder_X.fit_transform(X[:,4])
X[:,5] = labelencoder_X.fit_transform(X[:,5])
X[:,6] = labelencoder_X.fit_transform(X[:,6])
X[:,7] = labelencoder_X.fit_transform(X[:,7])
X[:,8] = labelencoder_X.fit_transform(X[:,8])
X[:, 10] = labelencoder_X.fit_transform(X[:, 10])
X[:, 11] = labelencoder_X.fit_transform(X[:, 11])
onehotencoder = OneHotEncoder(categorical_features = [11])
X = onehotencoder.fit_transform(X).toarray()
# ?-Need to check that if we can fit above statements in one line

#Avoiding Dummy variable trap
X=X[:,1:]

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
x_axis = np.linspace(1500000,15000000,109)
fig = plt.figure()
plt.plot(x_axis,y_test, color="red",  linewidth=2.5, linestyle="-")
plt.plot(x_axis,y_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Actual Vs Predicted ', fontsize=20)
# ?-Need to check better way to define X axis or how to keep both axis on same level

#Building the optimal model using Backward Elimination 
import statsmodels.formula.api as sm
'''
Following command will add array of one's as a single row to X array in the end

X=np.append(arr=X,values=np.ones((545,1)).astype(int),axis = 1)

To add the the one's in the begining just reverse the order in previous command
so that X as array added to one's array
'''
X=np.append(arr=np.ones((545,1)).astype(int),axis = 1,values=X)

X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 3, 4, 5,6,7,8,9,10,11,12,13]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 3, 5,6,7,8,9,10,11,12,13]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,2, 3,5,6,7,9,10,11,12,13]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [2,3,5,6,7,9,10,11,12,13]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

'''
Creating new train set with only X_opt  as this is the optimized data set
with this will calculate new predicted value 
With both predicted values and actual value will create a graph to compare them
'''
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