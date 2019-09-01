#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np


# In[45]:


train = pd.read_csv('zoo1.csv', index_col='animal_name')
train = train.sample(frac=1)
print(train.shape)
train.head()


# In[46]:


train2 = pd.read_csv('zoo2.csv', index_col='animal_name')
train2 = train2.sample(frac=1)
print(train2.shape)
train2.head()


# In[47]:


test = pd.read_csv('zoo3.csv', index_col='animal_name')
test = test.sample(frac=1)
print(test.shape)
test.head()


# In[48]:


example = test[['class_type']]
example['class_type'] = example.index.map(lambda x: len(x) % 7 + 1)
example.head()


# In[49]:


bool_cols = train.columns.values.tolist()
bool_cols.remove('legs')
bool_cols.remove('class_type')
bool_cols


# In[50]:


for df in [train, train2, test]:
    for col in bool_cols:
        df[col] = df[col].map({0: 'n', 1: 'y'}).astype(object)


# In[51]:


train.head()


# In[52]:


train.to_csv('zoo-train.csv')
train2.to_csv('zoo-train2.csv')

test.drop(['class_type'], axis=1).to_csv('zoo-test.csv')
test[['class_type']].to_csv('zoo-solution.csv')

example.to_csv('zoo-example.csv')


# In[53]:


get_ipython().system('head zoo-*.csv')


# In[ ]:




