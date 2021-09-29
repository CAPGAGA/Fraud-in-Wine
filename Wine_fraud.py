#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machines 
# ## Exercise
# 
# ## [Fraud in Wine](https://en.wikipedia.org/wiki/Wine_fraud)
# 
# Wine fraud relates to the commercial aspects of wine. The most prevalent type of fraud is one where wines are adulterated, usually with the addition of cheaper products (e.g. juices) and sometimes with harmful chemicals and sweeteners (compensating for color or flavor).
# 
# Counterfeiting and the relabelling of inferior and cheaper wines to more expensive brands is another common type of wine fraud.
# 
# <img src="wine.jpg">
# 
# ## Project Goals
# 
# A distribution company that was recently a victim of fraud has completed an audit of various samples of wine through the use of chemical analysis on samples. The distribution company specializes in exporting extremely high quality, expensive wines, but was defrauded by a supplier who was attempting to pass off cheap, low quality wine as higher grade wine. The distribution company has hired you to attempt to create a machine learning model that can help detect low quality (a.k.a "fraud") wine samples. They want to know if it is even possible to detect such a difference.
# 
# 
# Data Source: *P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties.
# In Decision Support Systems, Elsevier, 47(4):547-553, 2009.*
# 
# ---
# ---
# 
# Overall goal is to use the wine dataset shown below to develop a machine learning model that attempts to predict if a wine is "Legit" or "Fraud" based on various chemical features. Complete the tasks below to follow along with the project.**
# 
# ---
# ---

# Import

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("../DATA/wine_fraud.csv")


# In[3]:


df.head()


# Basic dataframe analsis

# In[7]:


df['quality'].unique()


# Creating a countplot that displays the count per category of Legit vs Fraud.

# In[8]:


sns.countplot(data=df,x='quality')


# Finding out if there is a difference between red and white wine when it comes to fraud.

# In[9]:


sns.countplot(data=df, x='type',hue='quality')


# Calculating what percentage of red wines are Fraud and percentage of white wines that are fraud

# In[26]:


print("Percentage of fraud in Red Wines:")
100* len(df[(df['type'] == 'red') & (df['quality'] == "Fraud")]) /len(df[(df['type'] == 'red') & (df['quality'] == "Legit")])


# In[27]:


print('Percentage of fraud in White Wines:')
100* len(df[(df['type'] == 'white') & (df['quality'] == "Fraud")]) /len(df[(df['type'] == 'white') & (df['quality'] == "Legit")])


# Calculating the correlation between the various features and the "quality" column.

# In[28]:


# CODE HERE
df['Fraud']= df['quality'].map({'Legit':0,'Fraud':1})


# In[29]:


df.corr()['Fraud']


# Creating a bar plot of the correlation values to Fraudlent wine.

# In[1]:


df.corr()['Fraud'][:-1].sort_values().plot(kind='bar')


# Creating a clustermap with seaborn to explore the relationships between variables.

# In[35]:


# CODE HERE
sns.clustermap(df.corr(),cmap='viridis')


# ----
# ## Machine Learning Model
# 
# Converting the categorical column "type" from a string or "red" or "white" to dummy variables:

# In[36]:


df['type'] = pd.get_dummies(df['type'],drop_first=True)


# Separate the data into X features and y target label ("quality" column)

# In[44]:


df.columns


# In[45]:


X=df.drop('quality',axis=1)
y=df['quality']


# Train/test split

# In[46]:


from sklearn.model_selection import train_test_split


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)


# Scaling data

# In[48]:


from sklearn.preprocessing import StandardScaler


# In[49]:


scaler = StandardScaler()


# In[50]:


scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)


# Creating SVM model

# In[51]:


# CODE HERE
from sklearn.svm import SVC


# In[52]:


svc_model = SVC(class_weight='balanced')


# Use a GridSearchCV to run a grid search for the best C and gamma parameters.

# In[53]:


from sklearn.model_selection import GridSearchCV


# In[54]:


param_grid = {'C' : [0.001,0.01,0.1,0.5,1],
             'gamma':['scale','auto']}


# In[55]:


grid_model = GridSearchCV(svc_model,param_grid)


# In[57]:


grid_model.fit(scaled_X_train,y_train)


# In[58]:


grid_model.best_params_


# In[61]:


y_pred = grid_model.predict(scaled_X_test)


# Display the confusion matrix and classification report for model.

# In[59]:


from sklearn.metrics import confusion_matrix, classification_report


# In[63]:


confusion_matrix(y_test,y_pred)


# In[65]:


print(classification_report(y_test,y_pred))

