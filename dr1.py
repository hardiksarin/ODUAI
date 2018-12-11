#Linear Regression Model

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the data set
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values

#spliting data set into test set and training set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size = 1/3, random_state = 0)

#fitting simple linear regressor into xtest
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train, y_train)

#predicting the test set
y_pred=regressor.predict(x_test)

#plotting the training graph
plt.scatter(x_train,y_train, color='red')
plt.plot(x_train, regressor.predict(x_train),color='blue')
plt.xlabel("Years of experiance")
plt.ylabel("Salary")
plt.show()
