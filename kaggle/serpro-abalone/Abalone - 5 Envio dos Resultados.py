#!/usr/bin/env python
# coding: utf-8

# ## Importação dos pacotes

# In[1]:


# importar pacotes necessários
import numpy as np
import pandas as pd


# In[2]:


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

# In[3]:


# carregar arquivo de dados de treino
train_data = pd.read_csv('abalone-train.csv', index_col='id')


# In[4]:


# carregar arquivo de dados de teste
test_data = pd.read_csv('abalone-test.csv', index_col='id')


# ## Transformações nos dados

# In[5]:


# gerar "one hot encoding" em atributos categóricos
train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)


# In[6]:


# encontrar e remover possíveis outliers
data = train_data
outliers = np.concatenate((
    data[(data['height'] < 0.01) | (data['height'] > 0.3)].index,
    data[(data['viscera_weight'] < 0.0001) | (data['viscera_weight'] > 0.6)].index
), axis=0)
train_data.drop(outliers, inplace=True)


# In[7]:


# realizar normalização nos dados numéricos contínuos
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
for data in [train_data, test_data]:
    data.loc[:,'length':'shell_weight'] = scaler.fit_transform(data.loc[:,'length':'shell_weight'])


# ## Seleção dos dados de treino e teste

# In[8]:


# definir dados de treino

X_train = train_data.drop(['rings'], axis=1) # tudo, exceto a coluna alvo
y_train = train_data['rings'] # apenas a coluna alvo

print('Forma dos dados de treino:', X_train.shape, y_train.shape)


# In[9]:


# definir dados de teste

X_test = test_data # tudo, já que não possui a coluna alvo

print('Forma dos dados de teste:', X_test.shape)


# ## Treinamento dos modelos e geração dos resultados 

# In[10]:


models = []

# Generalized Linear Models
models.append(('LinReg', LinearRegression(n_jobs=-1, fit_intercept=True, normalize=True)))
models.append(('LogReg', LogisticRegression(n_jobs=-1, random_state=42, multi_class='auto', C=1000, solver='sag')))
models.append(('OMP', OrthogonalMatchingPursuit(n_nonzero_coefs=7, fit_intercept=True, normalize=True)))
models.append(('PAR', PassiveAggressiveRegressor(random_state=42, C=0.2, fit_intercept=True, max_iter=1000, tol=0.001)))
models.append(('PP', Perceptron(random_state=42, penalty=None, alpha=1e-6, fit_intercept=True, max_iter=1000, tol=1e-3)))
models.append(('RANSAC', RANSACRegressor(random_state=42, min_samples=0.75)))
models.append(('Ridge', Ridge(random_state=42, alpha=0.1, fit_intercept=True, normalize=False)))
models.append(('SGD', SGDRegressor(random_state=42, alpha=1e-4, fit_intercept=True, penalty='l1', tol=1e-3)))
models.append(('TSR', TheilSenRegressor(random_state=42, n_jobs=-1, fit_intercept=True)))

# Decision Trees
models.append(('DTR', DecisionTreeRegressor(random_state=42, max_depth=4, min_samples_split=0.25)))

# Gaussian Processes
models.append(('GPR', GaussianProcessRegressor(random_state=42, alpha=0.01, normalize_y=True)))

# Kernel Ridge Regression
models.append(('KRR', KernelRidge(alpha=0.1)))

# Naïve Bayes
models.append(('GNB', GaussianNB(var_smoothing=0.001)))

# Nearest Neighbors
models.append(('kNN', KNeighborsRegressor(n_jobs=-1, n_neighbors=13, weights='distance')))

# Support Vector Machines
models.append(('SVM', SVR(gamma='auto', kernel='linear')))

# Neural network models
models.append(('MLP', MLPRegressor(random_state=42, max_iter=500,
                     activation='tanh', hidden_layer_sizes=(5,5,2), solver='lbfgs')))

# Ensemble Methods
models.append(('RFR', RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=100, max_depth=9)))
models.append(('GBR', GradientBoostingRegressor(random_state=42, learning_rate=0.05, n_estimators=100, subsample=0.8, max_depth=4, max_features=0.85)))
models.append(('ETR', ExtraTreesRegressor(random_state=42, n_jobs=-1, n_estimators=200, max_features=0.75)))
models.append(('BDTR', BaggingRegressor(random_state=42, n_jobs=-1, base_estimator=DecisionTreeRegressor(), max_features=0.75, n_estimators=75)))
models.append(('ABDTR', AdaBoostRegressor(random_state=42, n_estimators=200, base_estimator=DecisionTreeRegressor())))

# XGBoost
models.append(('XGBR', XGBRegressor(random_state=42, n_jobs=-1, learning_rate=0.1, n_estimators=50, max_depth=5, objective='reg:squarederror')))

# Voting
models.append(('VR', VotingRegressor(estimators=[
    ('MLP', MLPRegressor(random_state=42, max_iter=500, activation='tanh', hidden_layer_sizes=(5,5,2), solver='lbfgs')),
    ('GPR', GaussianProcessRegressor(random_state=42, alpha=0.01, normalize_y=True)),
    ('GBR', GradientBoostingRegressor(random_state=42, learning_rate=0.05, n_estimators=100, subsample=0.8, max_depth=4, max_features=0.85))
], n_jobs=-1, weights=(2,1,1))))

models = []
models.append(('MLP', MLPRegressor(random_state=42, max_iter=500,
                     activation='logistic', hidden_layer_sizes=(50,), solver='lbfgs')))
models.append(('GPR', GaussianProcessRegressor(random_state=42, alpha=0.01, normalize_y=True)))
models.append(('XGBR', XGBRegressor(random_state=42, n_jobs=2, learning_rate=0.1,
                                    n_estimators=50, max_depth=5, objective='reg:squarederror')))
# In[11]:


get_ipython().system('mkdir submissions')


# In[12]:


sufixo_arquivo = '03jul'

for name, model in models:
    print(model, '\n')
    
    # treinar o modelo
    model.fit(X_train, y_train)
    
    # executar previsão usando o modelo
    y_pred = model.predict(X_test)
    
    # gerar dados de envio (submissão)
    submission = pd.DataFrame({
      'id': X_test.index,
      'rings': y_pred
    })
    submission.set_index('id', inplace=True)

    # gerar arquivo CSV para o envio
    filename = 'submissions/abalone-submission-p-%s-%s.csv' % (sufixo_arquivo, name.lower())
    submission.to_csv(filename)


# In[13]:


# verificar conteúdo dos arquivos gerados
get_ipython().system('head submissions/abalone-submission-p-*.csv')


# In[ ]:




