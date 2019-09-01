#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importar pacotes necessários
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# definir parâmetros extras
import warnings
warnings.filterwarnings("ignore")
sns.set(style="white", color_codes=True)


# In[3]:


# carregar arquivo de dados de treino
data = pd.read_csv('abalone-train.csv', index_col='id')

# mostrar alguns exemplos de registros
data.head()


# In[4]:


# quantos moluscos existem de cada idade?
data['rings'].value_counts()


# In[5]:


data.iloc[:,:-1].head()


# In[6]:


# gerar gráfico de dispersão simples
data.plot(kind="scatter", x="length", y="diameter")


# In[7]:


# gerar gráfico de dispersão com histograma usando o pacote seaborn
sns.jointplot(x="length", y="diameter", data=data, size=8)


# In[8]:


# gerar gráfico similar usando a espécie na cor
sns.FacetGrid(data, hue="rings", size=8)    .map(plt.scatter, "length", "diameter")    .add_legend()


# In[9]:


# gerar um gráfico do tipo boxplot sobre uma característica individual
plt.figure(figsize=(16, 9))
sns.boxplot(x="rings", y="length", data=data)


# In[18]:


# gerar boxplot para cada uma das características por espécie
data.boxplot(by="rings", figsize=(16, 9))


# In[20]:


# gerar gráfico kde (densidade de kernel) sobre uma característica
sns.FacetGrid(data, hue="rings", size=8)    .map(sns.kdeplot, "length")    .add_legend()


# In[12]:


# gerar gráfico para analisar pares de características
sns.pairplot(data, hue="rings", size=3)


# In[13]:


# gerar gráfico em pares com kde nas diagonais
sns.pairplot(data, hue="rings", size=3, diag_kind="kde")


# In[27]:


# gerar mapa de calor com a correlação das características
plt.figure(figsize=(6,6))
corr = data.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, annot=True, fmt='.2f', mask=mask, cmap="YlGnBu")


# In[15]:


data.columns.values


# In[16]:


g = sns.FacetGrid(data, col="rings", margin_titles=True)
#bins = np.linspace(0, 60, 13)
g.map(plt.hist, "length", color="steelblue") #, bins=bins)


# In[17]:


sns.relplot(x="length", y="diameter", hue="rings", data=data)


# In[ ]:




