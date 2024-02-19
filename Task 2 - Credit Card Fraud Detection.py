#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries
import numpy as nm


# In[2]:


# loading train dataset
import pandas as pds
data=pds.read_csv("fraudTrain.csv")


# In[3]:


# printing the headings of each columns
data.columns


# In[4]:


# checking for null values
data.isna().sum()


# In[5]:


# changing the datatype of trans_date_time
import datetime
data["trans_date_trans_time"] = pds.to_datetime(data["trans_date_trans_time"])
data["dob"] = pds.to_datetime(data["dob"])
data


# In[6]:


# deleting the rows containing null values
data.dropna(ignore_index=True)


# In[7]:


# visualizing dataset
import matplotlib.pyplot as mltp
exit_counts = data["is_fraud"].value_counts()
mltp.figure(figsize=(9, 4))
mltp.subplot(1, 2, 1)  # Subplot for the pie chart
mltp.pie(exit_counts, labels=["No", "YES"], autopct="%0.0f%%")
mltp.title("is_fraud Counts")
mltp.tight_layout()  # Adjust layout to prevent overlapping
mltp.show()


# In[8]:


# train the model
features=['amt','lat','long','city_pop','unix_time','merch_lat','merch_long']
X1=data[features]
y1=data['is_fraud']


# In[9]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X1,y1)


# In[10]:


# reading test dataset
test_data=pds.read_csv("fraudTest.csv")
test_data


# In[11]:


test_data.drop(columns=['Unnamed: 0','cc_num','first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num','trans_date_trans_time'],inplace=True)
test_data


# In[12]:


X_test = test_data[features]
Y_test = test_data["is_fraud"]


# In[13]:


# predicting the model
y_pred = model.predict(X_test)
y_pred


# In[14]:


# testing accuracy Score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_data['is_fraud'],y_pred)
accuracy


# In[15]:


# calculating confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)
cm


# In[16]:


# visualizing test set results
import matplotlib.pyplot as mltp
from sklearn.metrics import roc_curve,auc
fpr, tpr, thresholds = roc_curve(Y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# plotting auc-roc curve
mltp.figure(figsize=(6, 4))
mltp.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
mltp.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
mltp.xlabel('False Positive Rate')
mltp.ylabel('True Positive Rate')
mltp.title('Receiver Operating Characteristic (ROC) Curve')
mltp.legend(loc='lower right')
mltp.show()

