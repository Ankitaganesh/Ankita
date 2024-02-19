#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd
data = pd.read_csv("Churn_Modelling.csv")
data.head()


# In[3]:


data.info()


# In[4]:


data.isnull().sum()


# In[5]:


data.columns


# In[6]:


data = data.drop(['RowNumber', 'CustomerId', 'Surname'],axis=1)


# In[7]:


data.head()


# In[8]:


data = pd.get_dummies(data,drop_first = True)
data.head()
data = data.astype(int)


# In[9]:


data.head()


# In[10]:


data['Exited'].value_counts()


# In[11]:


import matplotlib.pyplot as mtp
import seaborn as sns
mtp.figure(figsize =(8,6))
sns.countplot(x='Exited',data = data)


# In[12]:


X = data.drop('Exited',axis=1)
y = data['Exited']


# In[13]:


from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import precision_score, recall_score, f1_score


# In[14]:


from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.1, random_state=42)
print('Training Shape: ', X_train.shape)
print('Testing Shape: ', X_test.shape)


# In[15]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled


# In[16]:


#Logistic regression
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as LR, LogisticRegression as LRC
threshold = 0.5
y_train_classified = [1 if value > threshold else 0 for value in y_train]
LR = LRC()
LR.fit(X_train_scaled, y_train_classified)


# In[17]:


from sklearn.metrics import r2_score
y_test_classified = [1 if value > threshold else 0 for value in y_test]
ac1 = LR.score(X_test_scaled, y_test_classified)
print("Model Accuracy:", ac1)


# In[18]:


#support vector machine
from sklearn import svm
svm = svm.SVC()
svm.fit(X_train_scaled, y_train_classified)


# In[19]:


from sklearn.metrics import accuracy_score
ac2 = svm.score(X_test_scaled, y_test_classified)
print("Model Accuracy:", ac2)


# In[20]:


#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
rfc = RandomForestClassifier()
rfc.fit(X_train_scaled, y_train_classified)


# In[21]:


from sklearn.metrics import accuracy_score
ac3 = rfc.score(X_test_scaled, y_test_classified)
print("Model Accuracy:", ac3)


# In[22]:


#KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier as KNC
KNN = KNC()
KNN.fit(X_train_scaled, y_train_classified)


# In[23]:


from sklearn.metrics import accuracy_score
ac4 = KNN.score(X_test_scaled, y_test_classified)
print("Model Accuracy:", ac4)


# In[24]:


total_performance = pd.DataFrame({
    'Model':['LR','SVM','Random Forest','KNN'],
    'ACC':[ac1,
           ac2,
           ac3,
           ac4
          ]
})
total_performance

