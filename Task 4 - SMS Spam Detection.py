#!/usr/bin/env python
# coding: utf-8

# In[35]:


#importing libraries
import numpy as np


# In[37]:


import chardet
import pandas as pd

# # Detect the encoding of the file
# with open("spam.csv", 'rb') as file:
#     result = chardet.detect(file.read())
#     encoding = result['encoding']

# Read the CSV file with the detected encoding
data1 = pd.read_csv("spam.csv", encoding = 'latin-1')
data1.head(10)


# In[38]:


data1.shape


# In[39]:


data1 = data1.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)
data1.head(10)


# In[40]:


data1['v2'] = data1['v2'].str.lower()


# In[41]:


data1.v1.value_counts()


# In[42]:


data1["length"]=data1['v2'].apply(len)


# In[43]:


data1.head()


# In[44]:


import matplotlib.pyplot as mtp
import seaborn as sns


# In[45]:


mtp.figure(figsize=(6, 4))
sns.countplot(x='v1', data=data)
mtp.title('Distribution of Labels')
mtp.xlabel('Labels')
mtp.ylabel('Count')
mtp.show()


# In[47]:


mtp.figure(figsize=(8, 6))
mtp.hist(data1['length'], bins=30, color='green', edgecolor='black', alpha=0.7)
mtp.title('Distribution of Lengths')
mtp.xlabel('Message Length')
mtp.ylabel('Frequency')
mtp.grid(True)
mtp.show()


# In[48]:


data1.loc[:,'v1'] = data.v1.map({'ham':0,'spam':1})


# In[49]:


data1.head()


# In[50]:


from sklearn.feature_extraction.text import CountVectorizer as CV
count = CV()

v2 = count.fit_transform(data1['v2'])


# In[51]:


input = ["hello,how are you"]
text = count.fit_transform(data1['v2'],input)


# In[52]:


x = data1["v1"]
y = data1["v2"]


# In[53]:


from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(v2,x,test_size=0.2,random_state=42)


# In[54]:


print(x_train.shape)
print(x_test.shape)

input = v2[5571]


# In[55]:


x_train=x_train.astype(int)
y_train=y_train.astype(int)
x_test=x_test.astype(int)
y_test=y_test.astype(int)


# In[56]:


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(x_train,y_train)


# In[57]:


y_pred = classifier.predict(x_test)
y_pred


# In[58]:


print(type(y_test), type(y_pred))


# In[59]:


from sklearn.metrics import accuracy_score 
accuracy = accuracy_score(y_test,y_pred)
accuracy


# In[60]:


from sklearn.metrics import precision_score
precision = precision_score(y_test,y_pred)
precision


# In[61]:


from sklearn.metrics import recall_score
recall = recall_score(y_test,y_pred)
recall


# In[62]:


from sklearn.metrics import f1_score,confusion_matrix
f1 = f1_score(y_test,y_pred)
f1


# In[63]:


print(f"Accuracy score: {accuracy}")
print(f"Precision score: {precision}")
print(f"recall score: {recall}")
print(f"F1 score: {f1}")


# In[64]:


cm = confusion_matrix(y_test,y_pred)
cm


# In[65]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data["v2"])
X.toarray()


# In[66]:


def spam_msg_classifier(message):
    vectorized_msg = vectorizer.transform([message])
    if classifier.predict(vectorized_msg) == 1:
        print("Spam!!!!")
    else:
        print("Not Spam!!")


# In[67]:


spam_msg_classifier("Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...")


# In[68]:


spam_msg_classifier("Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's")


# In[69]:


spam_msg_classifier("You are a winner U have been specially selected 2 receive å£1000 or a 4* holiday (flights inc) speak to a live operator 2 claim 0871277810910p/min (18+) ")


# In[70]:


spam_msg_classifier("Goodo! Yes we must speak friday - egg-potato ratio for tortilla needed! ")

