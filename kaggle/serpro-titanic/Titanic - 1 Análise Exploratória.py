#!/usr/bin/env python
# coding: utf-8

# ## Importação dos pacotes

# In[2]:


# importar pacotes necessários
import numpy as np
import pandas as pd


# In[3]:


# definir parâmetros extras
pd.set_option('precision', 4)
pd.set_option('display.max_columns', 100)


# ## Carga dos dados

# In[23]:


# carregar arquivo de dados de treino
data = pd.read_csv('titanic-train.csv', index_col='person')

# mostrar alguns exemplos de registros
data.head()


# In[24]:


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


# sumário das características textuais
data.describe(include=['O']).T


# In[14]:


# quais as correlações entre as características numéricas?
corr = pd.get_dummies(data, columns=['survived', 'sex', 'embarked']).corr()
corr


# In[20]:


# quais as correlações mais expressivas entre as variáveis?
corr[corr != 1][abs(corr) > 0.05].dropna(how='all', axis=1).dropna(how='all', axis=0)


# In[21]:


data.groupby('survived').mean()


# In[29]:


data2 = data[['pclass', 'survived', 'sex', 'sibsp', 'parch']]
data2.survived = data2.survived.map({'yes': 1, 'no': 0})
data2.head()


# In[30]:


# existe correlação entre sobrevivência e classe social?
data2[['pclass', 'survived']].groupby(['pclass'], as_index=False).mean().  sort_values(by='survived', ascending=False)


# In[32]:


# existe correlação entre sobrevivência e sexo?
data2[['sex', 'survived']].groupby(['sex'], as_index=False).mean().  sort_values(by='survived', ascending=False)


# In[33]:


# existe correlação entre sobrevivência e número de irmãos?
data2[['sibsp', 'survived']].groupby(['sibsp'], as_index=False).mean().  sort_values(by='survived', ascending=False)


# In[34]:


# existe correlação entre sobrevivência e número de pais/filhos?
data2[['parch', 'survived']].groupby(['parch'], as_index=False).mean().  sort_values(by='survived', ascending=False)


# In[41]:


data[data.cabin.isnull() == False][['cabin', 'survived']].head(20).    sort_values(by='cabin', ascending=True)


# In[42]:


data['cabin'].value_counts()


# In[47]:


# extrair deque e número do quarto a partir da cabine
data['deck'] = data['cabin'].str[:1]
data['room'] = data['cabin'].str.extract("([0-9]+)", expand=False)


# In[48]:


data.head(20)


# In[50]:


# existe correlação da sobrevivência com o deque?
corr = pd.get_dummies(data, columns=['survived', 'deck']).corr()
corr


# In[ ]:




