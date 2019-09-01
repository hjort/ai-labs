#!/usr/bin/env python
# coding: utf-8

# ## Importação dos pacotes

# In[1]:


# importar pacotes necessários
import numpy as np
import pandas as pd


# In[2]:


# importar pacotes usados na medição da precisão
from sklearn.metrics import confusion_matrix


# ## Carga dos dados enviados (submetidos)

# In[3]:


# carregar arquivo de dados enviados
submission = pd.read_csv('iris-submission.csv', index_col='Id')

# mostrar alguns exemplos de registros
submission.head()


# In[7]:


y_pred = submission.Species


# ## Carga dos dados reais (esperados)

# In[4]:


# carregar arquivo de dados esperados
real_data = pd.read_csv('iris-solution.csv', index_col='Id')

# mostrar alguns exemplos de registros
real_data.head()


# In[5]:


y_real = real_data.Species


# ## Verificação dos dados enviados contra os dados reais

# In[14]:


# gerar e exibir matriz de confusão
cm = confusion_matrix(y_real, y_pred)
print('Matriz de Confusão:\n', cm)


# In[15]:


submission['Expected'] = y_real 
submission['Correct'] = (y_pred == y_real)


# In[18]:


incorretos = submission[submission.Correct == False]
print("Quantidade de itens incorretos: %d de %d (%.2f%%)" % (
    len(incorretos), len(submission), len(incorretos) / len(submission) * 100.0
))


# In[19]:


# exibir quais foram os itens classificados incorretamente
incorretos.head()


# In[ ]:




