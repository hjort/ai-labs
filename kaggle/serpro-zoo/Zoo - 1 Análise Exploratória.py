#!/usr/bin/env python
# coding: utf-8

# ## Importação dos pacotes

# In[1]:


# importar pacotes necessários
import numpy as np
import pandas as pd


# In[2]:


# definir parâmetros extras
pd.set_option('precision', 3)
pd.set_option('display.max_columns', 100)


# ## Carga dos dados de entrada

# ### Dados originais de treino

# In[3]:


# carregar arquivo de dados de treino
data = pd.read_csv('zoo-train.csv', index_col='animal_name')

# mostrar alguns exemplos de registros
data.head()


# In[4]:


# quantas linhas e colunas existem?
data.shape


# ### Dados adicionais de treino

# In[5]:


# carregar arquivo de dados de treino
data2 = pd.read_csv('zoo-train2.csv', index_col='animal_name')

# mostrar alguns exemplos de registros
data2.head()


# In[6]:


# quantas linhas e colunas existem?
data2.shape


# ### Unir dados de treinamento

# In[7]:


# unir ambos os dados de treinamento
data = data.append(data2)

# mostrar tamanho
print(data.shape)

# mostrar alguns exemplos de registros
data.tail()


# ## Analisar dados de treinamento

# In[8]:


# quais são as colunas e respectivos tipos de dados?
data.info()


# In[9]:


# existem colunas com dados nulos?
data[data.columns[data.isnull().any()]].isnull().sum()


# ## Transformações nos dados

# In[10]:


# classe do animal deve ser uma categoria
data['class_type'] = data['class_type'].astype('category')


# In[11]:


# atributos devem ser convertidos para 0 e 1

objcols = data.select_dtypes(['object']).columns
print(objcols)

data[objcols] = data[objcols].astype('category')
for col in objcols:
    data[col] = data[col].cat.codes


# In[12]:


data.info()


# In[13]:


data.tail()


# ## Outras análises nos dados

# In[14]:


# sumário estatístico das características numéricas
data.describe()


# In[15]:


# quais as correlações entre as características numéricas?
data.corr()


# In[22]:


# show variable correlation which is more than 0.7 (positive or negative)
corr = data.corr()
corr[corr != 1][abs(corr) > 0.7].dropna(how='all', axis=1).dropna(how='all', axis=0)


# In[23]:


data.groupby('class_type').mean()


# ## Gravar dados consolidados

# In[16]:


# gravar arquivo CSV consolidado
data.to_csv('zoo-train-all.csv')


# In[17]:


# carregar arquivo de dados de treino
data = pd.read_csv('zoo-train-all.csv', index_col='animal_name')
data['class_type'] = data['class_type'].astype('category')

# mostrar alguns exemplos de registros
data.head()


# In[ ]:




