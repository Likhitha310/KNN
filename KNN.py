#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression


# # Import the dataset
# 
# Link: https://raw.githubusercontent.com/omairaasim/machine_learning/master/project_11_k_nearest_neighbor/iphone_purchase_records.csv

# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/omairaasim/machine_learning/master/project_11_k_nearest_neighbor/iphone_purchase_records.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.Gender.value_counts()


# In[8]:


df.loc[df['Purchase Iphone']==1,"Gender"].value_counts()


# # Spliting of data

# In[9]:


df.head(2)


# In[10]:


X = df.iloc[:,:-1]
y = df.iloc[:,-1]


# # Label Encoding

# In[11]:


from sklearn.preprocessing import LabelEncoder


# In[12]:


enc = LabelEncoder()


# In[13]:


X.Gender = enc.fit_transform(X.Gender)


# In[14]:


X


# In[15]:


X.info()


# # Spliting the data into sets

# In[16]:


skf = StratifiedKFold(n_splits=5)


# In[17]:


for train_index,test_index in skf.split(X,y):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# # Feature Scaling

# In[18]:


scale = StandardScaler()


# In[19]:


X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)


# # Model selection

# In[20]:


log = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=5)


# # Training the model

# In[21]:


log.fit(X_train, y_train)
knn.fit(X_train, y_train)


# # Test the model

# In[22]:


y_log_pred = log.predict(X_test)


# In[23]:


y_knn_pred = knn.predict(X_test)


# In[24]:


newdf = pd.DataFrame({"Actual":y_test, "Predicted":y_knn_pred})


# In[25]:


newdf.head()


# In[26]:


confusion_matrix(y_test, y_knn_pred)


# In[28]:


from sklearn.metrics import accuracy_score


# In[29]:


accuracy_score(y_test, y_log_pred)


# In[30]:


lis = [i for i in range(2,101) if i%2==0]


# In[31]:


acc=[]
dic = {}
for i in lis:
  knn = KNeighborsClassifier(n_neighbors=i)
  knn.fit(X_train, y_train)
  y_knn_pred = knn.predict(X_test)
  acc.append(accuracy_score(y_test,y_knn_pred))
  # dic[i] = accuracy_score(y_test,y_knn_pred)

print(max(acc))
# 0.8875 = 89%
# 0.875 = 88%

