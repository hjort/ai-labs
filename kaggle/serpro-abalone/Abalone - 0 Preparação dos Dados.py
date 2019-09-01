#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# carregar dados
data = pd.read_csv('abalone.csv', sep=',')

# embaralhar dados
data = data.sample(frac=1)

# mostrar quantidade de linhas e colunas
print(data.shape)

# mostrar exemplos de dados
data.head()


# In[3]:


# renomear coluna índice
data.index.names = ['id']

# aumentar a precisão da coluna alvo
data.rings = round(data.rings + 1e-3, 2)
# In[4]:


data['rings'].describe()


# In[5]:


# exibir dados
data.head()


# In[6]:


# dividir os dados de treino e teste

divisao = int(data.shape[0] * 2 / 3)

train = data[:divisao]
test = data[divisao:]

print(train.shape, test.shape)


# In[7]:


train.head()


# In[8]:


test.head()


# In[9]:


example = test[['rings']]
example['rings'] = example.index.map(lambda x: int(x % 29 + 1))
example.head()


# In[10]:


example.describe()


# In[11]:


train.to_csv('abalone-train.csv')

test.drop(['rings'], axis=1).to_csv('abalone-test.csv')
test[['rings']].to_csv('abalone-solution.csv')

example.to_csv('abalone-example.csv')


# In[12]:


get_ipython().system('head abalone-*.csv')


# In[ ]:




