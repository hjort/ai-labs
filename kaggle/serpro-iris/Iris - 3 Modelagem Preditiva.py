#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importar pacotes necessários
import numpy as np
import pandas as pd


# In[2]:


# definir parâmetros extras
#pd.set_option('precision', 2)
pd.set_option('display.max_columns', 100)


# In[3]:


# carregar arquivo de dados de treino
data = pd.read_csv('iris-train.csv', index_col='Id')

# mostrar alguns exemplos de registros
data.head()


# In[4]:


# importar pacotes usados na seleção do modelo e na medição da precisão
from sklearn.model_selection import train_test_split

# importar os pacotes necessários para os algoritmos de classificação
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[5]:


# definir dados de entrada

X = data.drop(['Species'], axis=1) # tudo, exceto a coluna alvo
y = data['Species'] # apenas a coluna alvo

print('Forma dos dados originais:', X.shape, y.shape)


# In[6]:


# separarar dados para fins de treino (70%) e de teste (30%)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print('Forma dos dados separados:', X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[7]:


X_train.head()


# In[8]:


y_train.head()


# In[9]:


# A) Support Vector Machine (SVM)

model = SVC()

model.fit(X_train, y_train)
score = model.score(X_test, y_test) * 100

print(model, '\nScore:', score, '%')


# In[10]:


# B) Logistic Regression

model = LogisticRegression()

model.fit(X_train, y_train)
score = model.score(X_test, y_test) * 100

print(model, '\nScore:', score, '%')


# In[11]:


# C) Decision Tree

model = DecisionTreeClassifier()

model.fit(X_train, y_train)
score = model.score(X_test, y_test) * 100

print(model, '\nScore:', score, '%')


# In[12]:


# D) K-Nearest Neighbours

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)
score = model.score(X_test, y_test) * 100

print(model, '\nScore:', score, '%')


# In[ ]:




