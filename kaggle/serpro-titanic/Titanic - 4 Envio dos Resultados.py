#!/usr/bin/env python
# coding: utf-8

# ## Importação dos pacotes

# In[33]:


# importar pacotes necessários
import numpy as np
import pandas as pd


# In[34]:


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

# importar os pacotes necessários para os algoritmos de regressão
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
# ## Carga dos dados de entrada

# In[168]:


# carregar arquivo de dados de treino
train_data = pd.read_csv('titanic-train.csv', index_col='person')


# In[169]:


# carregar arquivo de dados de teste
test_data = pd.read_csv('titanic-test.csv', index_col='person')


# ## Transformações nos dados

# In[170]:


train_data.head()


# In[171]:


for data in [train_data, test_data]:
    print(data.shape)
    data.drop(['name', 'ticket', 'cabin', 'home_destination'], axis=1, inplace=True)
    
    data['age'].fillna(round(data.age.mean()), inplace=True)
    data['embarked'].fillna(data.embarked.mode()[0], inplace=True)
    data.fillna('0', inplace=True)
    
    #data.dropna(how='any', inplace=True)
    print(data.shape)

for data in [train_data, test_data]:
    print(data.isnull().sum())
# In[173]:


# gerar "one hot encoding" em atributos categóricos
cols = ['pclass', 'sex', 'embarked']
train_data = pd.get_dummies(train_data, columns=cols)
test_data = pd.get_dummies(test_data, columns=cols)


# In[174]:


# realizar normalização nos dados numéricos contínuos
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
for data in [train_data, test_data]:
    data.loc[:,'age':'fare'] = scaler.fit_transform(data.loc[:,'age':'fare'])


# In[175]:


train_data.head()


# ## Seleção dos dados de treino e teste

# In[176]:


# definir dados de treino

X_train = train_data.drop(['survived'], axis=1) # tudo, exceto a coluna alvo
y_train = train_data['survived'] # apenas a coluna alvo

print('Forma dos dados de treino:', X_train.shape, y_train.shape)


# In[177]:


# definir dados de teste

X_test = test_data # tudo, já que não possui a coluna alvo

print('Forma dos dados de teste:', X_test.shape)


# In[178]:


X_train.head()


# In[179]:


X_test.head()


# ## Treinamento dos modelos e geração dos resultados 

# In[180]:


models = []
models.append(('LR', LogisticRegression(random_state=42, solver='lbfgs', multi_class='auto', max_iter=500, C=100)))
models.append(('DT', DecisionTreeClassifier(random_state=42, criterion='gini', max_depth=11)))
models.append(('KNN', KNeighborsClassifier(n_neighbors=1)))
models.append(('SVM', SVC(random_state=42, C=10, gamma=0.1, kernel='rbf')))
models.append(('RF', RandomForestClassifier(random_state=42, max_features='auto', n_estimators=10)))
models.append(('SGD', SGDClassifier(random_state=42, max_iter=100, tol=0.1)))
models.append(('NN', Perceptron(random_state=42, max_iter=100, tol=0.01)))
models.append(('NB', GaussianNB(priors=None, var_smoothing=1e-08)))
models.append(('LSVM', LinearSVC(random_state=42, max_iter=1000, C=10)))
models.append(('ABDT', AdaBoostClassifier(DecisionTreeClassifier(random_state=42), n_estimators=5)))
models.append(('GB', GradientBoostingClassifier(random_state=42, max_depth=3)))
models.append(('MLP', MLPClassifier(random_state=42, solver='lbfgs', alpha=0.1, hidden_layer_sizes=(15,))))
models.append(('LDA', LinearDiscriminantAnalysis(solver='svd')))


# In[181]:


get_ipython().system('mkdir submissions')


# In[182]:


sufixo_arquivo = '03jul'

for name, model in models:
    print(model, '\n')
    
    # treinar o modelo
    model.fit(X_train, y_train)
    
    # executar previsão usando o modelo
    y_pred = model.predict(X_test)
    
    # gerar dados de envio (submissão)
    submission = pd.DataFrame({
      'person': X_test.index,
      'survived': y_pred
    })
    submission.set_index('person', inplace=True)

    # gerar arquivo CSV para o envio
    filename = 'submissions/titanic-submission-p-%s-%s.csv' % (sufixo_arquivo, name.lower())
    submission.to_csv(filename)


# In[183]:


# verificar conteúdo dos arquivos gerados
get_ipython().system('head submissions/titanic-submission-p-*.csv')


# In[ ]:




