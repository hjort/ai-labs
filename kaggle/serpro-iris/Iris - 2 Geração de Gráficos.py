#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importar pacotes necessários
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# definir parâmetros extras
import warnings
warnings.filterwarnings("ignore")
sns.set(style="white", color_codes=True)


# In[3]:


# carregar arquivo de dados de treino
data = pd.read_csv('iris-train.csv', index_col='Id')

# mostrar alguns exemplos de registros
data.head()


# In[4]:


# quantos registros existem de cada espécie?
data['Species'].value_counts()


# In[5]:


# gerar gráfico de dispersão simples
data.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")


# In[6]:


# gerar gráfico de dispersão com histograma usando o pacote seaborn
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=data, size=5, )


# In[7]:


# gerar gráfico similar usando a espécie na cor
sns.FacetGrid(data, hue="Species", size=5)    .map(plt.scatter, "SepalLengthCm", "SepalWidthCm")    .add_legend()


# In[8]:


# gerar um gráfico do tipo boxplot sobre uma característica individual
sns.boxplot(x="Species", y="PetalLengthCm", data=data)


# In[9]:


# gerar boxplot para cada uma das características por espécie
data.boxplot(by="Species", figsize=(12, 6))


# In[10]:


# gerar gráfico kde (densidade de kernel) sobre uma característica
sns.FacetGrid(data, hue="Species", size=6)    .map(sns.kdeplot, "PetalLengthCm")    .add_legend()


# In[11]:


# gerar gráfico para analisar pares de características
sns.pairplot(data, hue="Species", size=3)


# In[12]:


# gerar gráfico em pares com kde nas diagonais
sns.pairplot(data, hue="Species", size=3, diag_kind="kde")


# In[13]:


# gerar mapa de calor com a correlação das características
sns.heatmap(data.corr(), annot=True, cmap='cubehelix_r')


# In[ ]:




