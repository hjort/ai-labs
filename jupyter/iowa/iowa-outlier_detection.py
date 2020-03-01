#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv("./input/train.csv") 
test = pd.read_csv("prediction_training.csv").drop('Id',axis=1,inplace=False)
origin = pd.DataFrame(train['SalePrice'])


# In[3]:


dif = np.abs(test-origin) > 12000


# In[4]:


idx = dif[dif['SalePrice']].index.tolist()


# In[5]:


train.drop(train.index[idx],inplace=True)


# In[6]:


train.shape


# In[7]:


idx


# In[ ]:




