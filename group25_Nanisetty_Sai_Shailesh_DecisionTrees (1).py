#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Run this cell
#Importing necessary libraries 
import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt
import json
ans=[0]*5


# In[2]:


#Import the dataset and define the feature as well as the target datasets / columns 
df=pd.read_csv('zoo.csv')
#We drop the animal names since this is not a good feature to split the data on. 
df=df.drop('animal_name',axis=1)
df


# In[3]:


#Write a function to find the entropy on a split "target_col"
def entropy(target_col):
    ans=0
    for i in target_col.unique():
        p=sum(target_col==i)/len(target_col)
        if(p>0):
            ans-=p*(np.log(p)/np.log(2))
    return ans


# In[4]:


#Find the entropy of all the features in the dataset
#Save all the feature names in an array "feature names"
feature_names=['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone', 
               'breathes','venomous','fins','legs','tail','domestic','catsize']
for i in feature_names:
    print(entropy(df[i]))


# In[5]:


#Find the entropy of the feature "toothed"
ans[0]=entropy(df['toothed'])
ans[0]


# In[6]:


#Write a function to calculate Information Gain on a split attribute and a target column
def InfoGain(data,split_attribute_name,target_name="class"):       
    #Calculate the entropy of the total dataset  
    original_entropy=entropy(data[target_name])
    #Calculate the values and the corresponding counts for the split attribute   
    values=data[split_attribute_name].unique()
    sub=0
    for i in values:
        split=data[data[split_attribute_name]==i]
        #Calculate the weighted entropy  
        sub+=split.shape[0]/data.shape[0]*entropy(split[target_name])
    #Calculate the information gain  
    return original_entropy-sub


# In[7]:


#Find the information gain having split attribute "hair" and the target feature name "milk"
ans[1]=InfoGain(df,"hair","milk")
ans[1]


# In[8]:


#Find the Info gain having "milk" as the split attribute and all the other features as target features one at a time
for i in feature_names:
    if i!="milk":
        print(i+" - "+str(InfoGain(df,"milk",i)))


# In[9]:


#Import Decision Tree Classifier from sklearn 
from sklearn.tree import DecisionTreeClassifier
#Split the given data into 80 percent training data and 20 percent testing data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df[feature_names],df['class_type'],test_size=0.2,random_state=16)


# In[10]:


#Fit the given data
tree=DecisionTreeClassifier(random_state=16)
tree.fit(X_train,y_train)


# In[11]:


#Make a prediction on the test data and return the percentage of accuracy
y_pred=tree.predict(X_test)
ans[2]=tree.score(X_test,y_test)*100
ans[2]


# In[12]:


#Run this cell to visualize the decision tree
from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
export_graphviz(tree, out_file=dot_data, feature_names=feature_names,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[14]:


#Use sklearn to make a classification report and a confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
clf_report=classification_report(y_test,y_pred,output_dict=True)
cnf_matrix=confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))
print(cnf_matrix)


# In[15]:


#Find the recall,f1-score for class type '3'
ans[3]=[clf_report['3']['recall'],clf_report['3']['f1-score']]
ans[3]


# In[16]:


#Calculate Mean Absolute Error,Mean Squared Error and Root Mean Squared Error
from sklearn.metrics import mean_absolute_error,mean_squared_error
mae=mean_absolute_error(y_test, y_pred)
mse=mean_squared_error(y_test, y_pred)
rmse=np.sqrt(mse)


# In[17]:


#Find the mean absolute error and root mean square error, save then in a list [mae,rmse]
ans[4]=[mae,rmse]
ans[4]


# In[18]:


##do not change this code
import json
ans = [str(item) for item in ans]

filename = "shaileshnanisetty05@gmail.com_Nanisetty_Sai_Shailesh_DecisionTrees"

# Eg if your name is Saurav Joshi and email id is sauravjoshi123@gmail.com, filename becomes
# filename = sauravjoshi123@gmail.com_Saurav_Joshi_LinearRegression


# ## Do not change anything below!!
# - Make sure you have changed the above variable "filename" with the correct value. Do not change anything below!!

# In[19]:


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




