#!/usr/bin/env python
# coding: utf-8

# In[154]:


# Do not make any changes in this cell
# Simply execute it and move on

import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
ans = [0]*8


# In[155]:


# Simply execute this cell and move on

#Import the dataset and define the feature as well as the target datasets / columns  
dataset = pd.read_csv('heart.csv')  
#We drop the 'trestbps','chol','thalach','oldpeak' as they have numerical values  
dataset=dataset.drop('trestbps',axis=1) 
dataset=dataset.drop('chol',axis=1) 
dataset=dataset.drop('thalach',axis=1) 
dataset=dataset.drop('oldpeak',axis=1) 
dataset


# In[156]:


#These are the meanings of above features in the dataset
'''
age
sex
chest pain type (4 values)
resting blood pressure
serum cholestoral in mg/dl
fasting blood sugar > 120 mg/dl
resting electrocardiographic results (values 0,1,2)
maximum heart rate achieved
exercise induced angina
oldpeak = ST depression induced by exercise relative to rest
the slope of the peak exercise ST segment
number of major vessels (0-3) colored by flourosopy
thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
'''


# In[157]:


# Divide the age feature  into groups 
# Group the data based on age (<30 , range(30,40) , range(40,50) ,  range(50,60) , range(60,70) , >=70 )
#              = [ -1 ,     0        ,     1        ,         2     ,          3   ,   4   ]

# Write code here
dataset.loc[dataset['age'] < 30, 'age'] = -1
dataset.loc[dataset['age'] >= 70, 'age'] = 4
dataset.loc[dataset['age'] > 60, 'age'] = 3
dataset.loc[dataset['age'] > 50, 'age'] = 2
dataset.loc[dataset['age'] > 40, 'age'] = 1
dataset.loc[dataset['age'] > 30, 'age'] = 0


# In[ ]:


# QUESTION -1 :- (1mark)
#       Bayes theorm 
#       Find the following from the above data set 
#           - Find P("cp"=1,"thal"=2,"slope"=2 / Y=1) i.e.find the probability of ( "cp"=1 and "thal"=2 and "slope"=2 ) given that "target"=1
#       Assign your answer to ans[-1]


# In[161]:


#Write code here
df=dataset[dataset['target']==1]
y1=df.shape[0]
cp1=df['cp'].value_counts()[1]
thal2=df['thal'].value_counts()[2]
slope2=df['slope'].value_counts()[2]
ans[-1]=cp1*thal2*slope2/y1**3


# In[ ]:


#Write your answer here
ans[0]=0.544


# In[ ]:


# QUESTION -2 :- (1mark)
#       Find the prior distribution on the whole above data set , i.e. P(Y=1) and P(Y=0)
#       Enter value of P(Y=1) in ans[0]


# In[163]:


# Write your code here
#P(Y)
#Prior distribution
y=dataset.shape[0]
y0=y-y1
py_1=y1/y
ans[0]=py_1
ans[0]


# In[ ]:


# Enter your answer here
ans[1]=[0.08695,0.65942,0.25363]


# In[ ]:


# QUESTION -3:- (3marks)
#         Find the class conditional distribution on the above data set i.e. find P(X/Y)
#         Note: If class conditional probability = 0 assign 0.00000000000000001 (a low value) 
#         Find the class conditional probability of the feature "slope"= 0,1,2 when "target"=0 i.e. 
# P(X(slope)=i/Y=0) where i=0,1,2
#         Assign the answer to ans[1] in form of a list as [P(X(slope)=0/Y=0) , P(X(slope)=1/Y=0) ,
# P(X(slope)=2/Y=0)]


# In[166]:


#Write your code below
#Class conditional distribution
#P(x/y)

df0=dataset[dataset['target']==0]
slope0=df0['slope'].value_counts()[0]
slope1=df0['slope'].value_counts()[1]
slope2=df0['slope'].value_counts()[2]
ans[1]=[slope0/y0,slope1/y0,slope2/y0]
ans[1]


# In[ ]:


# Assign your answer here
ans[2]=


# In[ ]:


# QUESTION - 4:- (3marks)
#         Find the posterior distribution of the above dataset
#         i.e. Find P(Y/X) 
#         For X={"age":-1,'sex': 0,'cp':0 ,'fbs':1,'restecg':1,'exang':1,'slope':2,'ca':1,'thal':1}
#             - Find the label value (1 or 0) and also the respective confidence also the value of 
#               k in {  P(Y/X)=k*P(X1/Y)*P(X2/Y)*P(X3/Y)*P(X4/Y)......*P(Xn/Y)*P(Y) for n features   }
#         eg: if the answer is Y=0 with confidence =0.89 and k= 2100.8
#             your answer should be ans[2]=[0,0.89,2100.8]


# In[112]:


#Write code here
# posterior distribution
#P(y/x)

#when Y=1
_1age_neg1=df['age'].value_counts()[-1]
_1sex0=df['sex'].value_counts()[0]
_1cp0=df['cp'].value_counts()[0]
_1fbs1=df['fbs'].value_counts()[1]
_1restecg1=df['restecg'].value_counts()[1]
_1exang1=df['exang'].value_counts()[1]
_1thal1=df['thal'].value_counts()[1]
_1ca1=df['ca'].value_counts()[1]
_1slope2=df['slope'].value_counts()[2]

#when Y=0
_0age_neg1=0
_0sex0=df0['sex'].value_counts()[0]
_0cp0=df0['cp'].value_counts()[0]
_0fbs1=df0['fbs'].value_counts()[1]
_0restecg1=df0['restecg'].value_counts()[1]
_0exang1=df0['exang'].value_counts()[1]
_0thal1=df0['thal'].value_counts()[1]
_0ca1=df0['ca'].value_counts()[1]
_0slope2=df0['slope'].value_counts()[2]

p_y1_x=(_1age_neg1*_1sex0*_1cp0*_1fbs1*_1restecg1*_1exang1*_1thal1*_1ca1*_1slope2/y1**9)*y1/y
p_y0_x=0 #because _0age_neg1=0

prob_y1_x=p_y1_x/(p_y1_x+p_y0_x)
prob_y1_x
k=1/p_y1_x

ans[2]=[1,1,86583642.22]


# In[ ]:


#Write your answers here
ans[3]=8


# In[115]:


# Splitting the data for fitting a library naive bayes model from sklearn
# Use train_test_split to split the data

# Split the data into train and test (train-90% and test-10%)
# Strictly use (randon_state = 42) in train_test_split ,so that your answer can be evaluated

# Write your code here 
X=dataset[['age','sex','cp','fbs','restecg','thal','exang','slope','ca']]
Y=dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


# In[ ]:


# QUESTION - 5 :- (2marks)
#         Import the  GaussianNB  model from sklearn and find the no of wrong predictions on the testing set
# i.e. train and fit the model on the training set and predict the output if the heart disease exists or not 
#         Compare the predicted and the testing labels and enter the no.of wrongly predicted lables in ans[3]   


# In[153]:


#Importing the Gaussian naive bayes classifier model from sklearn

from sklearn.naive_bayes import GaussianNB

#Write your code below

clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

n=0
i=0

for x in y_test:
    if x!=y_pred[i]:
        n=n+1
    i=i+1
n    


# In[ ]:


#Write your answers here

ans[4]=


# In[2]:


import json
ans = [str(item) for item in ans]

filename = "sauravjoshi123@gmail.com_Saurav_Joshi_NaiveBayes"

# Eg if your name is Saurav Joshi and email id is sauravjoshi123@gmail.com, filename becomes
# filename = sauravjoshi123@gmail.com_Saurav_Joshi_NaiveBayes


# ## Do not change anything below!!
# - Make sure you have changed the above variable "filename" with the correct value. Do not change anything below!!

# In[3]:


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




