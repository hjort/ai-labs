#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# carregar dados
data = pd.read_csv('data/titanic.csv', sep=',')

# embaralhar dados
data = data.sample(frac=1)

# mostrar quantidade de linhas e colunas
print(data.shape)

# mostrar exemplos de dados
data.head()


# In[3]:


# renomear coluna índice
data.index.names = ['person']


# In[4]:


# renomear outras colunas
data.rename(columns={'home.dest': 'home_destination'}, inplace=True)


# In[5]:


# remover colunas "boat" e body"
data.drop(['boat', 'body'], axis=1, inplace=True)


# In[6]:


# alterar valor da coluna alvo
data['survived'] = data['survived'].map({0: 'no', 1: 'yes'})


# In[7]:


data.info()


# In[8]:


data.describe()


# In[9]:


# exibir dados
data.head()


# In[10]:


# dividir os dados de treino e teste

divisao = int(data.shape[0] * 2 / 3)

train = data[:divisao]
test = data[divisao:]

print(train.shape, test.shape)


# In[11]:


train.head()


# In[12]:


test.head()


# In[13]:


example = test[['survived']]
example['survived'] = example.index.map(lambda x: 'no' if (x % 2 == 0) else 'yes')
example.head()


# In[14]:


example.describe()


# In[15]:


train.to_csv('titanic-train.csv')

test.drop(['survived'], axis=1).to_csv('titanic-test.csv')
test[['survived']].to_csv('titanic-solution.csv')

example.to_csv('titanic-example.csv')


# In[16]:


get_ipython().system('head titanic-*.csv')


# In[ ]:




