# -*- coding: utf-8 -*-
"""group25_Nanisetty_Sai_Shailesh_RidgeRegression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ww4KDANNAdJP5KIHTztYSVg0Fro1uw1Q
"""

# Do not make any changes in this cell
# Simply execute it and move on
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
ans = [0]*8

# The exercise uses Boston housing dataset which is an inbuilt dataset of scikit learn.
# Run the cell below to import and get the information about the data.
# Do not make any changes in this cell.
# Simply execute it and move on
from sklearn.datasets import load_boston
boston=load_boston()
boston

# Creating a dataframe
# Do not make any changes in this cell
# Simply execute it and move on
boston_df=pd.DataFrame(boston['data'], columns=boston['feature_names'])
boston_df['target'] = pd.DataFrame(boston['target'])
boston_df

# Question 1: Find the mean of the "target" values in the dataframe (boston_df)  
#             Assign the answer to ans[0] 
#             eg. ans[0] = 24.976534890123 (if mean obtained = 24.976534890123)

# Your Code: Enter your Code below
target_mean=boston_df['target'].mean()

#1 mark
ans[0] = target_mean
ans[0]

# Just to get a look into distribution of data into datasets
# Plot a histogram for boston_df
li=boston_df.columns
for col in li:
  fig, ax = plt.subplots(figsize =(10, 7))
  ax.hist(boston_df[col])
  ax.legend(col)
  # Show plot
  plt.show()
###END###

"""**Splitting the data using train_test_split from sklearn library**"""

# Import machine learning libraries  for train_test_split

from sklearn.model_selection import train_test_split

# Split the data into X and Y
X=boston_df.iloc[:,0:13]
Y=boston_df.iloc[:,13]
# Spliting our data further into train and test (train-90% and test-10%)
# Use (randon_state = 42) in train_test_split, so that your answer can be evaluated
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.1,random_state=42)

"""**LINEAR REGRESSION**"""

# Question 2: Find mean squared error on the test set and the linear regression intercept(b)  
#             Assign the answer to ans[0] in the form of a list 
#             eg. ans[1] = [78.456398468,34.276498234098] 
#                  here , mean squared error             = 78.456398468
#                         linear regression intercept(b) = 34.276498234098

# Fit a linear regression model on the above training data and find MSE over the test set.
# Your Code: Enter your Code below
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error
reg = LinearRegression()
reg.fit(X_train,Y_train)
print(f'Accuracy we obtain from the model is {reg.score(X_train,Y_train)*100:.2f}%')
pred=reg.predict(X_test)
mean_squared_error=metrics.mean_squared_error(Y_test,pred)

res=[]
res.append(mean_squared_error)
res.append(reg.intercept_)

# 2 marks
ans[1] = res
ans[1]

"""**RIDGE REGRESSION**"""

# Question 3: For what value of lambda (alpha)(in the list[0.5,1,5,10,50,100]) will we have least value of the mean squared error of testing set 
#             Take lambda (alpha) values as specified i.e. [0.5,1,5,10,50,100]
#             Assign the answer to ans[2]  
#             eg. ans[1] = 5  (if  lambda(alpha)=5)

# Your Code: Enter your Code below
from sklearn.linear_model import Ridge
from sklearn import metrics
from sklearn.metrics import mean_squared_error
mse={}
for val in [0.5,1,5,10,50,100]:
  Ridgereg = Ridge(alpha=val)
  Ridgereg.fit(X_train,Y_train)
  print(f'Accuracy we obtain from the model is {Ridgereg.score(X_train,Y_train)*100:.2f}%')
  pred=Ridgereg.predict(X_test)
  mean_squared_error=metrics.mean_squared_error(Y_test,pred)
  mse[val]=mean_squared_error
mse

#1 mark
ans[2] = 1
ans[2]

# Question 4: Find mean squared error on the test set and the Ridge regression intercept(b)
#             Use the lamba(alpha) value obtained from question-3 
#             Assign the answer to ans[3] in the form of a list 
#             eg. ans[3] = [45.456398468,143.276498234098] 
#                  here , mean squared error             = 45.456398468
#                         Ridge regression intercept(b) = 143.276498234098

# Your Code: Enter your Code below
Ridgereg = Ridge(alpha=ans[2])
Ridgereg.fit(X_train,Y_train)
print(f'Accuracy we obtain from the model is {Ridgereg.score(X_train,Y_train)*100:.2f}%')
pred=Ridgereg.predict(X_test)
mean_squared_error=metrics.mean_squared_error(Y_test,pred)

res=[]
res.append(mean_squared_error)
res.append(Ridgereg.intercept_)

# 2 marks
ans[3] = res
ans[3]

# Plot the coefficient of the features( CRIM , INDUS , NOX ) with respective to  the lambda values specified [0.5,1,5,10,50,100]
# Enter your code below
#CRIM=(Index=0) feature, INDUS=(Index=2) feature,NOX=(Index=4) feature in the dataset
CRIM=[]
INDUS=[]
NOX=[]
Lambda=[0.5,1,5,10,50,100]
for val in Lambda:
  Ridgereg=Ridge(alpha=val)
  Ridgereg.fit(X_train,Y_train)
  pred=Ridgereg.predict(X_test)
  CRIM.append(Ridgereg.coef_[0])
  INDUS.append(Ridgereg.coef_[2])
  NOX.append(Ridgereg.coef_[4])
#CRIM
plt.figure()
plt.plot(Lambda,CRIM,label='CRIM')
plt.plot(Lambda,INDUS,label='INDUS')
plt.plot(Lambda,NOX,label='NOX')
plt.legend()
plt.show()

"""**LASSO REGRESSION**"""

# Question 5: For lambda (alpha)=1 find the lasso regression intercept and the test set mean squared error
#             Assign the answer to ans[4] in the form of a list 
#             eg. ans[4] = [35.456398468,14.276498234098] 
#                  here , mean squared error             = 35.456398468
#                         lasso regression intercept(b) = 14.276498234098

# Your Code: Enter your Code below
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.metrics import mean_squared_error
Lassoreg = Lasso(alpha=1)
Lassoreg.fit(X_train,Y_train)
print(f'Accuracy we obtain from the model is {Lassoreg.score(X_train,Y_train)*100:.2f}%')
pred=Lassoreg.predict(X_test)
mean_squared_error=metrics.mean_squared_error(Y_test,pred)

res=[]
res.append(mean_squared_error)
res.append(Lassoreg.intercept_)

#2 mark
ans[4] = res
res

# Question 6: Find the most  important feature  in the data set i.e. which feature coefficient is further most non zero if lambda is increased gradually
#             let CRIM=1,	ZN=2, INDUS=3,	CHAS=4,	NOX=5,	RM=6,	AGE=7,	DIS=8,	RAD=9,	TAX=10,	PTRATIO=11,	B=12,	LSTAT=13
#              eg. if your answer is "CHAS"
#                   then your answer should be ans[5]=4

# Your Code: Enter your Code below
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.metrics import mean_squared_error
for val in [0.5,1,5,10,50,100]:
  Lassoreg = Lasso(alpha=val)
  Lassoreg.fit(X_train,Y_train)
  #print(f'Accuracy we obtain from the model is {Lassoreg.score(X_train,Y_train)*100:.2f}%')
  pred=reg.predict(X_test) 
  mean_squared_error=metrics.mean_squared_error(Y_test,pred)
  print(Lassoreg.coef_)

#2 marks
#Feature 'TAX' is further most non zero if lambda is increased gradually and hence is the most important feature
ans[5] = 10
ans[5]

"""Run the below cell only once u complete answering all the above answers 

"""

##do not change this code
import json
ans = [str(item) for item in ans]

filename = "shaileshnanisetty05@gmail.com_Nanisetty_Sai_Shailesh_RidgeRegression"

# Eg if your name is Saurav Joshi and email id is sauravjoshi123@gmail.com, filename becomes
# filename = sauravjoshi123@gmail.com_Saurav_Joshi_LinearRegression

"""## Do not change anything below!!
- Make sure you have changed the above variable "filename" with the correct value. Do not change anything below!!
"""

from importlib import import_module
import os
from pprint import pprint
findScore = import_module('findScore')
response = findScore.main(ans)
response['details'] = filename
with open(f'evaluation_{filename}.json', 'w') as outfile:
    json.dump(response, outfile)
pprint(response)
