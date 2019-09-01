#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('iris.csv', index_col='Id')
data.head()


# In[3]:


data = data.sample(frac=1)

print(data.shape)
data.head()


# In[4]:


train = data[:100]
test = data[100:]

print(train.shape, test.shape)


# In[5]:


train.head()


# In[6]:


test.head()


# In[7]:


species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

example = test[['Species']]
example['Species'] = example.index.map(lambda x: species[x % 3])
example.head()


# In[8]:


train.to_csv('iris-train.csv')
test.drop(['Species'], axis=1).to_csv('iris-test.csv')
test[['Species']].to_csv('iris-solution.csv')
example.to_csv('iris-example.csv')


# In[9]:


get_ipython().system('head *.csv')


# In[ ]:




