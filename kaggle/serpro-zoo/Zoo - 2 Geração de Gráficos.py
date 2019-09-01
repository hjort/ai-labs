#!/usr/bin/env python
# coding: utf-8

# ## Importação dos pacotes

# In[22]:


# importar pacotes necessários
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.utils import shuffle


# In[23]:


# definir parâmetros extras
import warnings
warnings.filterwarnings("ignore")
sns.set(style="white", color_codes=True)


# ## Carga dos dados de entrada (consolidados)

# In[33]:


# carregar arquivo de dados de treino
data = pd.read_csv('zoo-train-all.csv', index_col='animal_name')

# embaralhar linhas
#data = shuffle(data)
data = data.sample(frac=1)

# deixar coluna como categórica
data['class_type'] = data['class_type'].astype('category')

# mostrar alguns exemplos de registros
data.head()


# In[34]:


# 1-7 is Mammal, Bird, Reptile, Fish, Amphibian, Bug and Invertebrate
animal_type = ['Mammal', 'Bird', 'Reptile', 'Fish', 'Amphibian', 'Bug', 'Invertebrate']

data['class_name'] = data['class_type'].map(lambda x: animal_type[x-1])

data.iloc[:,-2:].head()


# In[35]:


# quantos registros existem de cada espécie?
data['class_type'].value_counts()


# In[37]:


sns.countplot(data['class_name'])


# In[39]:


data.legs.unique()


# In[46]:


# just curious which animal has 5 legs
data.loc[data['legs'] == 5][['class_type', 'class_name']]


# In[40]:


sns.countplot(data['legs'])

# gerar gráfico similar usando a espécie na cor
sns.FacetGrid(data, hue="class_type", size=5) \
   .map(plt.scatter, "hair", "toothed") \
   .add_legend()# gerar um gráfico do tipo boxplot sobre uma característica individual
sns.boxplot(x="class_type", y="eggs", data=data)# gerar boxplot para cada uma das características por espécie
data.boxplot(by="class_type", figsize=(12, 6))# gerar gráfico kde (densidade de kernel) sobre uma característica
sns.FacetGrid(data, hue="class_type", size=6) \
   .map(sns.kdeplot, "legs") \
   .add_legend()
# In[10]:


# gerar gráfico para analisar pares de características
#sns.pairplot(data, hue="class_type", size=3)


# In[11]:


# gerar gráfico em pares com kde nas diagonais
#sns.pairplot(data, hue="class_type", size=3, diag_kind="kde")


# In[38]:


# gerar mapa de calor com a correlação das características
plt.figure(figsize=(14,14))
sns.heatmap(data.corr(), annot=True, fmt='.2f')


# 

# In[42]:


data.groupby('class_name').mean()


# In[47]:


g = sns.FacetGrid(data, col="class_name")
g.map(plt.hist, "legs")
plt.show()


# In[ ]:




