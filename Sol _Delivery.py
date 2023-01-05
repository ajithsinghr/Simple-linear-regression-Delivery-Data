# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 15:58:34 2022

@author: ramav
"""
# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# file

df = pd.read_csv("D:\Assignments\simple linear regresssion\delivery_time.csv")
df.head()
df.shape
df.isnull().sum()

X= df[["Sorting Time"]]
Y = df[["Delivery Time"]]

# exploratry data analysis(EDA)

plt.scatter(X.iloc[:,0], Y,color="black")
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

plt.boxplot(X)
plt.boxplot(Y)


plt.hist(X,histtype="bar")
plt.hist(Y)

df.mean()
df.median()

# Model Fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
LR.intercept_ #Bo
LR.coef_ #B1

# Predict the value
Y_pred = LR.predict(X)
Y_pred

# Scatter Plot with Plot
plt.scatter (X.iloc[:,0],Y,color = 'red')
plt.plot (X.iloc[:,0],Y_pred,color = 'blue')
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

# R2
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(Y, Y_pred)
RMSE = np.sqrt(mse)
print("Root mean square error: ", RMSE.round(3))
print("Rsquare: ", r2_score(Y, Y_pred).round(3)*100)


#  RMSE is 2.792 and the R2 is  68.2


# Transformation
# Model 2
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(np.log(X),Y)
y1 = LR.predict(np.log(X))

mse= mean_squared_error(Y, y1)
RMSE=np.sqrt(mse).round(3)
print("Root mean square error: ", RMSE)
print("Rsquare: ", r2_score(Y, y1).round(3)*100)

plt.scatter (X.iloc[:,0],Y,color = 'red')
plt.plot (X.iloc[:,0],y1,color = 'blue')
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()


#  RMSE is 2.733 and the R2 is  69.5


# Model 3
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(np.sqrt(X),Y)
y1 = LR.predict(np.sqrt(X))

mse= mean_squared_error(Y, y1)
RMSE=np.sqrt(mse).round(3)
print("Root mean square error: ", RMSE)
print("Rsquare: ", r2_score(Y, y1).round(3)*100)

plt.scatter (X.iloc[:,0],Y,color = 'red')
plt.plot (X.iloc[:,0],y1,color = 'blue')
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()


#  RMSE is 2.732 and the R2 is 69.6


# Model 4
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X**2,Y)
y1 = LR.predict(X**2)

mse= mean_squared_error(Y, y1)
RMSE=np.sqrt(mse).round(3)
print("Root mean square error: ", RMSE)
print("Rsquare: ", r2_score(Y, y1).round(3)*100)

plt.scatter (X.iloc[:,0],Y,color = 'red')
plt.plot (X.iloc[:,0],y1,color = 'blue')
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()


#  RMSE is 3.011 and the R2 is 63.0


# Model 5

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X**3,Y)
y1 = LR.predict(X**3)

mse= mean_squared_error(Y, y1)
RMSE=np.sqrt(mse).round(3)
print("Root mean square error: ", RMSE)
print("Rsquare: ", r2_score(Y, y1).round(3)*100)

plt.scatter (X.iloc[:,0],Y,color = 'red')
plt.plot (X.iloc[:,0],y1,color = 'blue')
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()


#  RMSE is 3.253 and the R2 is 56.89

# Among the following models the best mode is model3 because it have rscore of 69.6 and the graph is bell shaped





