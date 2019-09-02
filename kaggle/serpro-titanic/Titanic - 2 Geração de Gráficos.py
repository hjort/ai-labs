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
data = pd.read_csv('titanic-train.csv', index_col='person')

# mostrar alguns exemplos de registros
data.head()


# In[4]:


# quantos passageiros sobreviveram e quantos não?
data['survived'].value_counts()


# In[6]:


# crianças tiveram maiores chances de sobrevivência?
g = sns.FacetGrid(data, col='survived')
g.map(plt.hist, 'age', bins=20)


# In[8]:


grid = sns.FacetGrid(data, col='pclass', hue='survived')
grid.map(plt.hist, 'age', alpha=.5, bins=20)
grid.add_legend()


# In[10]:


grid = sns.FacetGrid(data, col='embarked')
grid.map(sns.pointplot, 'pclass', 'survived', 'sex', palette='deep')
grid.add_legend()


# In[11]:


grid = sns.FacetGrid(data, row='embarked', col='survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'sex', 'fare', alpha=.5, ci=None)
grid.add_legend()


# In[ ]:





# In[8]:


# gerar gráfico de dispersão simples
data.plot(kind="scatter", x="age", y="fare")


# In[9]:


# gerar gráfico de dispersão com histograma usando o pacote seaborn
sns.jointplot(x="age", y="fare", data=data, size=8)


# In[10]:


# gerar gráfico similar usando a espécie na cor
sns.FacetGrid(data, hue="survived", size=8)    .map(plt.scatter, "age", "fare")    .add_legend()


# In[12]:


# gerar um gráfico do tipo boxplot sobre uma característica individual
plt.figure(figsize=(16, 9))
sns.boxplot(x="age", y="fare", data=data)


# In[13]:


# gerar boxplot para cada uma das características por espécie
data.boxplot(by="survived", figsize=(16, 9))


# In[14]:


# gerar gráfico kde (densidade de kernel) sobre uma característica
sns.FacetGrid(data, hue="survived", size=8)    .map(sns.kdeplot, "age")    .add_legend()


# In[15]:


# gerar gráfico para analisar pares de características
sns.pairplot(data, hue="survived", size=3)


# In[16]:


# gerar gráfico em pares com kde nas diagonais
sns.pairplot(data, hue="survived", size=3, diag_kind="kde")


# In[17]:


# gerar mapa de calor com a correlação das características
plt.figure(figsize=(6,6))
corr = data.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, annot=True, fmt='.2f', mask=mask, cmap="YlGnBu")


# In[18]:


data.columns.values


# In[19]:


g = sns.FacetGrid(data, col="survived", margin_titles=True)
#bins = np.linspace(0, 60, 13)
g.map(plt.hist, "age", color="steelblue") #, bins=bins)


# In[20]:


sns.relplot(x="age", y="fare", hue="survived", data=data)


# In[ ]:




