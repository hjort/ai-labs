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
data = pd.read_csv('weather-train.csv', index_col='date', parse_dates=['date'])

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


# qual o período de datas?
print(min(data.index.values), '=>', max(data.index.values))


# In[9]:


10.0 / data.shape[0]


# In[16]:


periodo = '2W' # quinzenal
#periodo = '7D' # semanal

minimas = data.resample(periodo).min()
maximas = data.resample(periodo).max()
medias = data.resample(periodo).mean()
desvios = data.resample(periodo).std()

minimas.rename(columns={'temperature': 'minima'}, inplace=True)
maximas.rename(columns={'temperature': 'maxima'}, inplace=True)
medias.rename(columns={'temperature': 'media'}, inplace=True)
desvios.rename(columns={'temperature': 'desvio'}, inplace=True)


# In[17]:


medias.info()


# In[18]:


temperaturas = minimas.merge(maximas, on='date').merge(medias, on='date').merge(desvios, on='date')


# In[19]:


temperaturas['media-dp'] = temperaturas.media - temperaturas.desvio
temperaturas['media+dp'] = temperaturas.media + temperaturas.desvio


# In[20]:


temperaturas.head(10).T


# In[22]:


#temperaturas[['media']].plot(figsize=(16,8))
temperaturas[['media', 'media-dp', 'media+dp']].plot(figsize=(16,8))


# In[ ]:




