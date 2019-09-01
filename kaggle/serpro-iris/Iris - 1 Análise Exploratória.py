#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importar pacotes necessários
import numpy as np
import pandas as pd


# In[2]:


# definir parâmetros extras
pd.set_option('precision', 2)
pd.set_option('display.max_columns', 100)


# In[3]:


# carregar arquivo de dados de treino
data = pd.read_csv('iris-train.csv', index_col='Id')

# mostrar alguns exemplos de registros
data.head()


# In[4]:


# quantas linhas e colunas existem?
data.shape


# In[5]:


# quais são as colunas e respectivos tipos de dados?
data.info()


# In[6]:


# sumário estatístico das características numéricas
data.describe()


# In[7]:


# existem colunas com dados nulos?
data[data.columns[data.isnull().any()]].isnull().sum()


# In[8]:


# quais as correlações entre as características numéricas?
data.corr()


# In[ ]:




