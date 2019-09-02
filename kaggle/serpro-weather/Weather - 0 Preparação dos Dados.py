#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# definir parâmetros extras
pd.set_option('precision', 4)
pd.set_option('display.max_columns', 100)


# In[3]:


# carregar dados
data1 = pd.read_csv('temperature.csv', sep=',', index_col='datetime', parse_dates=['datetime'])
data2 = pd.read_csv('humidity.csv', sep=',', index_col='datetime', parse_dates=['datetime'])
data3 = pd.read_csv('wind_speed.csv', sep=',', index_col='datetime', parse_dates=['datetime'])

# mostrar quantidade de linhas e colunas
print(data1.shape, data2.shape, data3.shape)

# mostrar exemplos de dados
data1.head()


# In[4]:


data1.info()


# In[5]:


data1.describe()


# In[6]:


# manter apenas uma cidade
cidade = 'Detroit'
data1 = data1[[cidade]]
data2 = data2[[cidade]]
data3 = data3[[cidade]]

# renomear colunas
data1.rename(columns={cidade: 'temperature'}, inplace=True)
data2.rename(columns={cidade: 'humidity'}, inplace=True)
data3.rename(columns={cidade: 'wind_speed'}, inplace=True)


# In[7]:


# converter temperatura de Kelvin para graus Celsius
data1['temperature'] = round(data1['temperature'].apply(lambda k: k - 273.15), 4)


# In[8]:


# mesclar todas em uma única tabela
data = data1.merge(data2, on='datetime').merge(data3, on='datetime').dropna(axis=0)


# In[9]:


# manter apenas a temperatura média diária no período das 9h às 17h
data = data[(data.index.hour >= 9) & (data.index.hour <= 17)].resample('D').mean()


# In[10]:


data.head()


# In[11]:


# incrementar temperatura em 10ºC ao longo de todo o período (5 anos)

data = data.reset_index()
incremento_diario = 10.0 / data.shape[0]

data['incremento'] = float(0)
data['incremento'] = data.apply(lambda x: x.index * incremento_diario)
data['temperature'] += data['incremento']

data.drop(['incremento'], axis=1, inplace=True)
data = data.set_index('datetime')


# In[12]:


data.head()


# In[13]:


# normalizar a chave para somente data
#data.index = data.index.normalize()
data.index.names = ['date']


# In[14]:


data.info()


# In[15]:


data.describe()


# In[16]:


# exibir dados
data.sample(5)


# In[17]:


data.head()


# In[18]:


# dividir os dados de treino e teste

divisao = int(data.shape[0] * 4 / 5)

train = data[:divisao]
test = data[divisao:]

print(train.shape, test.shape)


# In[19]:


train.head()


# In[20]:


test.head()


# In[21]:


example = test[['temperature']]

# gerar valores aleatórios de temperatura (mín: -30ºC, máx: 35ºC)
example['temperature'] = np.round(np.random.random(example.shape[0]) * 65 - 30, 4)

example.head()


# In[22]:


example.describe()

train.to_csv('weather-train.csv')

test.drop(['temperature'], axis=1).to_csv('weather-test.csv')
test[['temperature']].to_csv('weather-solution.csv')

example.to_csv('weather-example.csv')
# In[26]:


train[['temperature']].to_csv('weather-train.csv')

test.drop(['temperature', 'humidity', 'wind_speed'], axis=1).to_csv('weather-test.csv')
test[['temperature']].to_csv('weather-solution.csv')

example.to_csv('weather-example.csv')


# In[27]:


get_ipython().system('head weather-*.csv')


# In[ ]:




