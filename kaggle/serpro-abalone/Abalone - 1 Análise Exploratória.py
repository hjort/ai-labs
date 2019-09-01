#!/usr/bin/env python
# coding: utf-8

# ## Importação dos pacotes

# In[1]:


# importar pacotes necessários
import numpy as np
import pandas as pd


# In[2]:


# definir parâmetros extras
pd.set_option('precision', 4)
pd.set_option('display.max_columns', 100)


# ## Carga dos dados

# In[3]:


# carregar arquivo de dados de treino
data = pd.read_csv('abalone-train.csv', index_col='id')

# mostrar alguns exemplos de registros
data.head()


# In[4]:


# quantas linhas e colunas existem?
data.shape


# ## Análise dos dados

# In[5]:


# quais são as colunas e respectivos tipos de dados?
data.info()


# In[6]:


# existem colunas com dados nulos?
data[data.columns[data.isnull().any()]].isnull().sum()


# In[7]:


# sumário estatístico das características numéricas
data.describe().T


# In[8]:


# quais as correlações entre as características numéricas?
data.corr()


# In[9]:


# show variable correlation which is more than 0.6 (positive or negative)
corr = data.corr()
corr[corr != 1][abs(corr) > 0.6].dropna(how='all', axis=1).dropna(how='all', axis=0)


# In[10]:


data.groupby('rings').mean()


# In[11]:


numeric_feats = data.dtypes[data.dtypes != "object"].index
numeric_feats

from scipy.stats import skew
skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())) # compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
skewed_feats
# In[12]:


data.head(10).T

data[skewed_feats] = np.log1p(data[skewed_feats])data.head(10).Tdata = pd.get_dummies(data)
data = data.fillna(data.mean())
# In[13]:


data.isna().sum()


# In[14]:


data.sex.describe()


# In[15]:


data.head()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print(scaler.fit(data))

data_scaled = pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index)
data_scaled.head()
# In[ ]:




