#!/usr/bin/env python
# coding: utf-8

# # ***` Logistic Regression `***
# 
# ***` Import the Iris data which discusses about three species of flowers namely "Setosa","Verisicolor" and "Virginica" Your task is to build a logistic regression model to distinguish between two  of these speicies using features like "Sepal Length", "Sepal Width", "Petal Length" and "Petal Width"`***
# 
# `1)Write a sigmoid function and visualize the sigmoid function,by considering x in the range of (-10,10).`
# 
# `2)Plot impact of logloss for single forecasts (You can import log_loss from sklearn.metrics). Make predictions as 0 to 1 in 0.01 increments. (For example,yhat = [x*0.01 for x in range(0, 101)]).Evaluate predictions for a 0 true value.Plot a graph between y_hat and log_loss`
# 
# `3)Find the difference between minimum log loss for label 0 and label 1 [1.5 marks]`
# 
# `3)Import the Iris Data, and visualize the data to an idea about it.`
# 
# `4)Convert the char labels to numerical as logistic regression takes only 0's and 1's and then create new array of numerical labels.After following the procedure as mentioned in the comments , find the difference between means of sepal length of speices "Setosa"(label 0) and "Versicolor"(label 1).[1 marks]`
# 
# `5)Split the data in X,y and convert them into arrays`
# 
# `6)Use sklearn to split the data (**Important** Consider random_state=42 and test_size=0.2)and perform Logistic Regression`
# 
# `7)Find the weights and bias and save it in a list[5 marks]`
# 
# `8)Make a prediction on the test data.Find the accuracy of the prediction.[1 marks]`
# 
# `9)Also predict the species of the flower whose sepal length=4.9cm	sepal width=4cm	petal length=1.2cm	petal width=0.4cm and return the Species of the data.[1.5 marks]`
# 

# In[63]:


# Run this cell
# import important libraries library
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
get_ipython().run_line_magic('matplotlib', 'inline')
ans = [0]*5


# # ***`Importing and Visualizing Data`***
# 
# 

# In[64]:


#Sigmoid Function
def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[65]:


#Visualize sigmoid function
#Create an array of x_val with values between -10 and 10 
x_val=np.linspace(-10,10,100)
#Find y_val, by using sigmoid function
y_val=sigmoid(x_val)
#Plot x_val,y_val and label the graph
plt.plot(x_val,y_val)
plt.xlabel('X')
plt.ylabel('Sigmoid(X)')
plt.show()


# In[66]:


# Plot impact of logloss for single forecasts
from sklearn.metrics import log_loss
# predictions as 0 to 1 in 0.01 increments
y_hat=[x*0.01 for x in range(0,101)]
# evaluate predictions for a 0 true value
loss_0=[log_loss([0],[x],labels=[0,1])for x in y_hat]
# evaluate predictions for a 1 true value
loss_1=[log_loss([1],[x],labels=[0,1])for x in y_hat]
# plot input to loss
plt.plot(y_hat,loss_0,label='true=0')
plt.plot(y_hat,loss_1,label='true=1')
plt.legend()
plt.show()


# In[67]:


#Find the difference between minimum log loss for label 0 and label 1 
ans[0]=min(loss_0)-min(loss_1)
ans[0]


# # ***`Processing the Data`***

# In[68]:


#Import the dataset of iris from datasets.load_iris()
iris=datasets.load_iris()
df=pd.DataFrame(iris.data,columns=['sepal_length','sepal_width','petal_length','petal_width'])


# In[69]:


#Look into the top 5 rows of data
df.head()


# In[70]:


#Visualize  the data using seaborn pairplot
sns.pairplot(df)


# In[71]:


# Convert char labels into numerical 
#import LabelEncoder which returns array of encoded labels
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
# Create new array of numerical labels
labelencoder.fit(iris.target)


# In[72]:


# Drop old labels(char) data 
target=pd.Series(iris.target,name='target')
# Substitute new labels(numerical) into data
data=pd.concat([df,target],axis=1)
data


# In[73]:


# Logistic regression only takes the data which has labels 0 and 1, so consider only data['labels']<2
# Considering Iris-setosa as "0" and Iris-versicolor as "1"
data=data[data['target']<2]


# In[74]:


#Find the difference between means of sepal length of speices "Setosa"(label 0) and "Versicolor"(label 1)
ans[1]=np.mean(data[data['target']==1]['sepal_length'])-np.mean(data[data['target']==0]['sepal_length'])
ans[1]


# # ***`Obtaining Weight Values`***

# In[75]:


# Split the data into X and y
X=data[['sepal_length','sepal_width','petal_length','petal_width']]
y=data['target']
y


# In[106]:


# Visualize X,y


# In[77]:


# Convert X,y into arrays
X=np.array(X)
y=np.array(y)


# In[78]:


#Using sklearn to split the data
from sklearn.model_selection import train_test_split
#Take the test size as 0.2 and random_state as 42
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[79]:


#Importing Necessary Libraries for Logistic Regression 
from sklearn.linear_model import LogisticRegression
#Building our model
model = LogisticRegression(random_state=42)
model.fit(X_train,y_train)
#Finding the parameter and bias


# In[80]:


#Printing the parameters and bias
print(model.intercept_,model.coef_)


# In[90]:


#Save parameters and bias [w1,w2,w3,w4,b] as one vector 
#i.e if the answer should be in a 1 dimensional list
ans[2]=[model.coef_[0][0],model.coef_[0][1],model.coef_[0][2],model.coef_[0][3],model.intercept_[0]]
ans[2]


# In[91]:


#Predicitng on our test data
model.predict(X_test)


# In[92]:


#Finding the accuracy
ans[3]=model.score(X_test,y_test)
ans[3]


# In[97]:


#Predict for the input [4.9,4,1.2,0.4] , save the answer ans[4] "Setosa" or "Versicolor"
model.predict([[4.9,4,1.2,0.4]])


# In[99]:


#The class of the input 
ans[4]="Setosa"
ans[4]


# In[100]:


import json
ans = [str(item) for item in ans]

filename = "oqaistanvir@gmail.com_Oqais_Tanvir_LogisticRegression"

# Eg if your name is Saurav Joshi and email id is sauravjoshi123@gmail.com, filename becomes
# filename = sauravjoshi123@gmail.com_Saurav_Joshi_LogisticRegression


# ## Do not change anything below!!
# - Make sure you have changed the above variable "filename" with the correct value. Do not change anything below!!

# In[101]:


from importlib import import_module
import os
from pprint import pprint

findScore = import_module('findScore')
response = findScore.main(ans)
response['details'] = filename
with open(f'evaluation_{filename}.json', 'w') as outfile:
    json.dump(response, outfile)
pprint(response)


# In[ ]:




