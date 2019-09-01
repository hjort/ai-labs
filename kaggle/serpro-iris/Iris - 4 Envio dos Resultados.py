#!/usr/bin/env python
# coding: utf-8

# ## Importação dos pacotes

# In[1]:


# importar pacotes necessários
import numpy as np
import pandas as pd


# In[2]:


# importar os pacotes necessários para os algoritmos de classificação
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ## Carga dos dados de entrada (treino e teste)

# In[3]:


# carregar arquivo de dados de treino
train_data = pd.read_csv('iris-train.csv', index_col='Id')

# mostrar alguns exemplos de registros
train_data.head()


# In[4]:


# carregar arquivo de dados de teste
test_data = pd.read_csv('iris-test.csv', index_col='Id')

# mostrar alguns exemplos de registros
test_data.head()


# In[5]:


# definir dados de treino

X_train = train_data.drop(['Species'], axis=1) # tudo, exceto a coluna alvo
y_train = train_data['Species'] # apenas a coluna alvo

print('Forma dos dados de treino:', X_train.shape, y_train.shape)


# In[6]:


# definir dados de teste

X_test = test_data # tudo, já que não possui a coluna alvo

print('Forma dos dados de teste:', X_test.shape)


# ## Transformações nos dados
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(np.concatenate((X_train.values, X_test.values), axis=0))

X_train = pd.DataFrame(scaler.transform(X_train.values), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test.values), columns=X_test.columns, index=X_test.index)from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(np.concatenate((X_train.values, X_test.values), axis=0))

X_train = pd.DataFrame(pca.transform(X_train.values), columns=['pc1','pc2','pc3'], index=X_train.index)
X_test = pd.DataFrame(pca.transform(X_test.values), columns=['pc1','pc2','pc3'], index=X_test.index)
# ## Treinamento do modelo preditivo

# In[7]:


# definir modelo a ser gerado
#model = KNeighborsClassifier(n_neighbors=3)
#model = SVC(random_state=42, C=1, gamma=0.001, kernel='linear')
#model = MLPClassifier(random_state=42, solver='lbfgs', alpha=1, hidden_layer_sizes=(15,))
#model = LogisticRegression(random_state=42, solver='lbfgs', multi_class='auto', max_iter=500, C=10)
#model = DecisionTreeClassifier(random_state=42, max_depth=5, criterion='entropy')
model = LinearDiscriminantAnalysis(solver='svd')
#model = RandomForestClassifier(random_state=42, max_features='auto', n_estimators=100)
#model = AdaBoostClassifier(DecisionTreeClassifier(random_state=42), n_estimators=1)
#model = GradientBoostingClassifier(random_state=42, max_depth=5)
#model = GaussianNB(priors=None, var_smoothing=1e-08)
#model = LinearSVC(random_state=42, max_iter=1000, C=1)

# definir sufixo
suffix = 'lda'

print(model)


# In[8]:


model.fit(X_train, y_train)


# ## Predição dos resultados e criação do arquivo de envio

# In[9]:


# executar previsão usando o modelo escolhido
y_pred = model.predict(X_test)

print('Exemplos de previsões:\n', y_pred[:10])


# In[10]:


# gerar dados de envio (submissão)

submission = pd.DataFrame({
  'Id': X_test.index,
  'Species': y_pred
})
submission.set_index('Id', inplace=True)

# mostrar dados de exemplo
#submission.head(10)


# In[11]:


# gerar arquivo CSV para o envio
arquivo = 'iris-submission-' + suffix + '.csv'
submission.to_csv(arquivo)


# In[12]:


# verificar conteúdo do arquivo gerado
get_ipython().system('head $arquivo')

