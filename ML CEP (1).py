#!/usr/bin/env python
# coding: utf-8

# In[2]:


#name : Mahin Rahim     and     Ahmed bilal
#cms :     56555                   56993
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


# In[19]:


calories_data= pd.read_csv('calories.csv')


# In[20]:


exercise_data = pd.read_csv('exercise.csv')


# In[21]:


exercise_data.head()


# In[22]:


calories_data.head()


# In[23]:


calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)


# In[24]:


calories_data.head()


# In[25]:


# checking the number of rows and columns
calories_data.shape


# In[26]:


# getting some informations about the data
calories_data.info()


# In[ ]:





# In[27]:


# checking for missing values
calories_data.isnull().sum()


# In[28]:


#DATA ANAYLSIS
# get some statistical measures about the data
calories_data.describe()


# In[31]:


#data visualization
sns.set()


# In[50]:


# plotting the gender column in count plot
sns.countplot(calories_data['Gender'])


# In[19]:


# finding the distribution of "Age" column
sns.distplot(calories_data['Age'])


# In[33]:


sns.distplot(calories_data['Height'])


# In[34]:


sns.distplot(calories_data['Weight'])


# In[35]:


correlation = calories_data.corr()


# In[24]:


# constructing a heatmap to understand the correlation

plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')


# In[36]:


calories_data.replace({"Gender":{'male':0,'female':1}}, inplace=True)


# In[37]:


calories_data.head()


# In[38]:


X = calories_data.drop(columns=['User_ID','Calories'], axis=1)
Y = calories_data['Calories']


# In[39]:


print(X)


# In[40]:


print(Y)


# In[41]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[42]:


print(X.shape, X_train.shape, X_test.shape)


# In[43]:


# loading the model
model = XGBRegressor()


# In[44]:


# training the model with X_train
model.fit(X_train, Y_train)


# In[45]:


test_data_prediction = model.predict(X_test)


# In[46]:


print(test_data_prediction)


# In[47]:


print(test_data_prediction)


# In[48]:


mae = metrics.mean_absolute_error(Y_test, test_data_prediction)


# In[49]:


print("Mean Absolute Error = ", mae)


# In[ ]:




