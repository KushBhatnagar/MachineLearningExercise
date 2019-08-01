# -*- coding: utf-8 -*-
"""
Created on Thu Aug 01 06:08:15 2019

@author: Kush
This Script will predict salary based on the number of year of exp. 
I have used Polynomial Regression as it suited the data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset=pd.read_csv('SalaryPred.csv')

#Polynomial LR Approach
X=dataset.iloc[:,0:1].values
y=dataset.iloc[:,1:2].values

#Fitting Polynomial Regression to the data set
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
lin_reg=LinearRegression()
lin_reg.fit(X_poly,y)

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X,lin_reg.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Predicted Vs Actual')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

y_pred2=lin_reg.predict(poly_reg.fit_transform([[6]]))
