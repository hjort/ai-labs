#!/usr/bin/env python
# coding: utf-8

# ## Importação dos pacotes

# In[1]:


# importar pacotes necessários
import numpy as np
import pandas as pd


# In[2]:


# definir parâmetros extras
pd.set_option('precision', 4)
pd.set_option('display.max_columns', 100)

import warnings
warnings.filterwarnings("ignore")


# ## Carga dos dados

# In[3]:


#prefixo_arquivos = ''
prefixo_arquivos = 'https://github.com/hjort/ai-labs/raw/master/kaggle/serpro-titanic/'


# In[4]:


# carregar arquivo de dados de treino
train_data = pd.read_csv(prefixo_arquivos + 'titanic-train.csv', index_col='person')


# In[5]:


# carregar arquivo de dados de teste
test_data = pd.read_csv(prefixo_arquivos + 'titanic-test.csv', index_col='person')


# In[6]:


# unir ambos os dados de treino e teste
data = pd.concat([train_data, test_data])

# mostrar alguns exemplos de registros
data.head()


# ## Transformações nos dados

# In[7]:


# transformar colunas textuais em categóricas
data['survived'] = data['survived'].map({'yes': 1, 'no': 0})


# In[8]:


# extrair títulos das pessoas a partir do nome
data['title'] = data['name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# exibir relação entre título e sexo
pd.crosstab(data['title'], data['sex']).T


# In[9]:


# agregar títulos incomuns
replacements = {
    'Miss': ['Mlle', 'Ms'],
    'Mrs': ['Mme'],
    'Rare': ['Lady', 'Countess', 'Capt', 'Col', 'Don', \
             'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
}
for k, v in replacements.items():
    data['title'] = data['title'].replace(v, k)
    
# exibir relação entre título e sexo
pd.crosstab(data['title'], data['sex']).T


# In[10]:


# categorizar os valores dos títulos
title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
data['title'] = data['title'].map(title_mapping)
data['title'] = data['title'].fillna(0)


# In[11]:


# categorizar os valores dos sexos
data['sex'] = data['sex'].map({'female': 1, 'male': 0}).astype(int)


# In[12]:


# preencher e categorizar os valores dos portos de embarque
data['embarked'].fillna(data.embarked.mode()[0], inplace=True)
data['embarked'] = data['embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)


# In[13]:


# preencher os valores da passagem
data['fare'].fillna(data.fare.mean(), inplace=True)


# In[14]:


# criar coluna com tamanho da família
data['fsize'] = data['parch'] + data['sibsp'] + 1


# In[15]:


# criar coluna indicando se estava sozinho
data['alone'] = 0
data.loc[data.fsize == 1, 'alone'] = 1


# In[16]:


# criar coluna contendo o deque
data['deck'] = data['cabin'].str[:1]
data['deck'] = data['deck'].fillna('N').astype('category')
data['deck'] = data['deck'].cat.codes


# In[17]:


# criar coluna contendo o número do quarto
data['room'] = data['cabin'].str.extract("([0-9]+)", expand=False)
data['room'] = data['room'].fillna(0).astype(int)


# In[18]:


data.head()


# ### Inferir idades faltantes dos passageiros

# In[19]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def evaluate_regression_model(model, X, y):
    kfold = KFold(n_splits=10, random_state=42)
    results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error', verbose=1)
    score = (-1) * results.mean()
    stddev = results.std()
    print(model, '\nScore: %.2f (+/- %.2f)' % (score, stddev))
    return score, stddev


# In[20]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

age_models = [
#    ('LR', LinearRegression(n_jobs=-1, fit_intercept=True, normalize=True)),
    ('GBR', GradientBoostingRegressor(random_state=42)),
    ('RFR', RandomForestRegressor(random_state=42)),
    ('XGB', XGBRegressor(random_state=42, objective='reg:squarederror')),
#    ('MLP', MLPRegressor(random_state=42, max_iter=500, activation='tanh',
#                         hidden_layer_sizes=(10,5,5), solver='lbfgs')),
#    ('GPR', GaussianProcessRegressor(random_state=42, alpha=0.01, normalize_y=True))
]


# In[21]:


# selecionar dados para o treino

cols = ['pclass', 'sibsp', 'parch', 'fare', 'title', 'age', 'alone']
#cols = ['pclass', 'sibsp', 'parch', 'fare', 'sex', 'embarked', 'title', 'age', 'alone']

data_age = data[cols].dropna()

X_age = data_age.drop(['age'], axis=1)
y_age = data_age['age']

data_age.head()


# In[22]:


data_age.corr()


# In[23]:


names = []
scores = []
lowest = 999
best_model = None

for name, model in age_models:
    
    score, stddev = evaluate_regression_model(model, X_age, y_age)
    names.append(name)
    scores.append(score)
    
    if score < lowest:
        best_model = model
        lowest = score


# In[24]:


results = pd.DataFrame({'Age Model': names, 'Score': scores})
results.sort_values(by='Score', ascending=True)


# In[25]:


age_model = best_model
age_model.fit(X_age, y_age)


# In[26]:


# preencher dados faltantes de idade a partir de uma regressão
data['age_pred'] = age_model.predict(data[cols].drop('age', axis=1))
data.loc[data.age.isnull(), 'age'] = data['age_pred']
data.drop('age_pred', axis=1, inplace=True)
data.head()


# In[27]:


# existem colunas com dados nulos?
data[data.columns[data.isnull().any()]].isnull().sum()

# criar colunas adicionais
data['ageclass'] = data['age'] * data['pclass']
data['pfare'] = data['fare'] / data['fsize']
# In[28]:


data.head()

# gerar "one hot encoding" em atributos categóricos
#cols = ['pclass', 'sex', 'embarked']
cols = ['embarked', 'pclass', 'title', 'deck']
data = pd.get_dummies(data, columns=cols)
# In[29]:


# realizar normalização nos dados numéricos contínuos
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#cols = ['age', 'fare']
cols = ['age', 'fare', 'parch', 'pclass', 'sibsp', 'fsize']
#cols = ['age', 'embarked', 'fare', 'parch', 'pclass', 'sex', 'sibsp', 'title', 'fsize', 'deck']
#cols = ['age', 'embarked', 'fare', 'parch', 'pclass', 'sex', 'sibsp', 'title', 'fsize', 'deck', 'room']

#for col in cols:
data.loc[:,cols] = scaler.fit_transform(data.loc[:,cols])


# In[30]:


data.head()


# ## Modelagem preditiva

# In[31]:


# importar os pacotes necessários para os algoritmos de classificação
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
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
from xgboost import XGBClassifier


# In[32]:


from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# avalia o desempenho do modelo, retornando o valor da precisão
def evaluate_classification_model(model, X, y):
    start = datetime.now()
    kfold = KFold(n_splits=10, random_state=42)
    results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy', verbose=1)
    end = datetime.now()
    elapsed = int((end - start).total_seconds() * 1000)
    score = 100.0 * results.mean()
    stddev = 100.0 * results.std()
    print(model, '\nScore: %.2f (+/- %.2f) [%5s ms]' % (score, stddev, elapsed))
    return score, stddev, elapsed


# In[33]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

# faz o ajuste fino do modelo, calculando os melhores hiperparâmetros
def fine_tune_model(model, params, X, y):
    print('\nFine Tuning Model:')
    print(model, "\nparams:", params)
    kfold = KFold(n_splits=10, random_state=42)
    grid = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=kfold, verbose=1)
    grid.fit(X, y)
    print('\nGrid Best Score: %.2f' % (grid.best_score_ * 100.0))
    print('Best Params:', grid.best_params_)
    return grid


# In[34]:


# definir dados de treino
train_data = data[data.survived.isnull() == False]

# selecionar atributos para o modelo
cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'title', 'fsize', 'alone', 'deck']
#cols = ['pclass', 'sex', 'age', 'fare', 'title', 'fsize', 'deck']
#cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'title', 'fsize', 'alone', 'deck', 'room']

X_train = train_data[cols]
y_train = train_data['survived']

print('Forma dos dados de treino:', X_train.shape, y_train.shape)


# In[35]:


train_data.corr()


# In[36]:


# definir dados de teste
test_data = data[data.survived.isnull()]

X_test = test_data[cols]

print('Forma dos dados de teste:', X_test.shape)


# In[37]:


names = []
models = []
scores = []
stddevs = []
times = []

def add_model_info(name, model, score, stddev, elapsed):
    names.append(name)
    models.append((name, model))
    scores.append(score)
    stddevs.append(stddev)
    times.append(elapsed)


# ## Avaliação e ajuste fino de cada modelo preditivo
# 
# -  https://scikit-learn.org/stable/modules/classes.html

# ### Generalized Linear Models

# In[38]:


model = LogisticRegression(random_state=42, solver='newton-cg', C=0.1, multi_class='auto', max_iter=500)

params = dict(
    solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    C=np.logspace(-3, 3, 7)
)
#fine_tune_model(model, params, X_train, y_train)

score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)
add_model_info('LR', model, score, stddev, elapsed)


# ### Decision Trees

# In[39]:


model = DecisionTreeClassifier(random_state=42, criterion='entropy', max_depth=6, min_samples_split=0.25)

#criterion=’mse’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
#min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, 
#min_impurity_decrease=0.0, min_impurity_split=None, presort=False

params = dict(
    criterion=['gini','entropy'],
    max_depth=[4, 6, 8, 10, 12, 14],
    min_samples_split=[0.25, 0.5, 0.75, 1.0]
)
#fine_tune_model(model, params, X_train, y_train)

score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)
add_model_info('DT', model, score, stddev, elapsed)


# ### Discriminant Analysis

# In[40]:


model = LinearDiscriminantAnalysis(solver='lsqr')

#solver=’svd’, shrinkage=None, priors=None,
#n_components=None, store_covariance=False, tol=0.0001

params = dict(
    solver=['svd', 'lsqr'] #, 'eigen']
)
#fine_tune_model(model, params, X_train, y_train)

score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)
add_model_info('LDA', model, score, stddev, elapsed)


# ### Naïve Bayes

# In[41]:


model = GaussianNB(priors=None, var_smoothing=1e-8)

#priors=None, var_smoothing=1e-09

params = dict(
    priors=[None],
    var_smoothing=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
)
#fine_tune_model(model, params, X_train, y_train)

score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)
add_model_info('NB', model, score, stddev, elapsed)


# ### Nearest Neighbors

# In[42]:


model = KNeighborsClassifier(n_neighbors=11, weights='uniform')

#n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’,
#metric_params=None, n_jobs=None

params = dict(
    n_neighbors=[1, 3, 5, 7, 9, 11, 13],
    weights=['uniform', 'distance']
)
#fine_tune_model(model, params, X_train, y_train)

score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)
add_model_info('KNN', model, score, stddev, elapsed)


# ### Support Vector Machines

# In[43]:


model = SVC(random_state=42, C=10, gamma=0.1, kernel='rbf')

#kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0, tol=0.001, C=1.0, 
#epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1

params = dict(
    C=[0.001, 0.01, 0.1, 1, 10, 100],
    gamma=[0.001, 0.01, 0.1, 1, 10, 100],
    kernel=['linear', 'rbf']
)
#fine_tune_model(model, params, X_train, y_train)

score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)
add_model_info('SVM', model, score, stddev, elapsed)


# ### Neural network models

# In[44]:


model = MLPClassifier(random_state=42, solver='lbfgs', alpha=1, hidden_layer_sizes=(100,), activation='logistic')
                
#hidden_layer_sizes=(100, ), activation=’relu’, solver=’adam’, alpha=0.0001, batch_size=’auto’, 
#learning_rate=’constant’, learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, 
#random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
#early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10

params = dict(
    alpha=[1,0.1,0.01,0.001,0.0001,0],
    hidden_layer_sizes=[(100,), (50,), (50,2), (5,5,2), (10,5,2)],
    activation=['identity', 'logistic', 'tanh', 'relu'],
    solver=['lbfgs', 'sgd', 'adam']
)
#fine_tune_model(model, params, X_train, y_train)

score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)
add_model_info('MLP', model, score, stddev, elapsed)


# ### Ensemble Methods

# In[45]:


model = AdaBoostClassifier(DecisionTreeClassifier(random_state=42), random_state=42, n_estimators=50)

#base_estimator=None, n_estimators=50, learning_rate=1.0,
#algorithm=’SAMME.R’, random_state=None

params = dict(
    n_estimators=[10, 25, 50, 100]
)
#fine_tune_model(model, params, X_train, y_train)

score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)
add_model_info('ABDT', model, score, stddev, elapsed)


# In[46]:


from sklearn.ensemble import BaggingClassifier

model = BaggingClassifier(random_state=42, n_estimators=100,
                          max_samples=0.25, max_features=0.8)

#base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0,
#bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False,
#n_jobs=None, random_state=None, verbose=0

params = dict(
    n_estimators=[10, 50, 100, 500],
    max_samples=[0.25, 0.5, 0.75, 1.0],
    max_features=[0.7, 0.8, 0.9, 1.0]
)
#fine_tune_model(model, params, X_train, y_train)

score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)
add_model_info('BC', model, score, stddev, elapsed)


# In[47]:


model = ExtraTreesClassifier(random_state=42, n_estimators=100, max_depth=7, 
                             min_samples_split=0.25, max_features='auto')

#n_estimators=’warn’, criterion=’gini’, max_depth=None, min_samples_split=2,
#min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, 
#max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 
#bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0,
#warm_start=False, class_weight=None

params = dict(
    n_estimators=[10, 50, 100, 500],
    max_depth=[None, 3, 7, 11],
    min_samples_split=[0.25, 0.5],
    max_features=['auto', 0.7, 0.85, 1.0]
)
#fine_tune_model(model, params, X_train, y_train)

score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)
add_model_info('ET', model, score, stddev, elapsed)


# In[48]:


model = GradientBoostingClassifier(random_state=42, n_estimators=100, max_features=0.75,
                                   max_depth=4, learning_rate=0.1, subsample=0.6)

#loss=’ls’, learning_rate=0.1, n_estimators=100, subsample=1.0, criterion=’friedman_mse’, min_samples_split=2,
#min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, 
#min_impurity_split=None, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, 
#max_leaf_nodes=None, warm_start=False, presort=’auto’, validation_fraction=0.1, n_iter_no_change=None, 
#tol=0.0001

params = dict(
    n_estimators=[100, 250, 500],
    max_features=[0.75, 0.85, 1.0],
    max_depth=[4, 6, 8, 10],
    learning_rate=[0.05, 0.1, 0.15],
    subsample=[0.4, 0.6, 0.8]
)
#fine_tune_model(model, params, X_train, y_train)

score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)
add_model_info('GB', model, score, stddev, elapsed)


# In[49]:


model = RandomForestClassifier(random_state=42, n_estimators=100, max_features='auto', max_depth=5)

#n_estimators=’warn’, criterion=’mse’, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
#min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, 
#min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, 
#verbose=0, warm_start=False

params = dict(
    n_estimators=[10, 50, 100, 500],
    max_features=['auto', 'sqrt', 'log2'],
    max_depth=[None, 3, 5, 7, 9, 11, 13]
)
#fine_tune_model(model, params, X_train, y_train)

score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)
add_model_info('RF', model, score, stddev, elapsed)


# ### Outros algoritmos

# #### XGBoost
# 
# - https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn

# In[50]:


model = XGBClassifier(max_depth=3, n_estimators=100)

#max_depth=3, learning_rate=0.1, n_estimators=100, verbosity=1, silent=None, objective='reg:squarederror',
#booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, 
#colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, 
#base_score=0.5, random_state=0, seed=None, missing=None, importance_type='gain'

params = dict(
    max_depth=[3, 5, 7, 9],
    n_estimators=[50, 75, 100, 200]
)
#fine_tune_model(model, params, X_train, y_train)

score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)
add_model_info('XGB', model, score, stddev, elapsed)


# In[ ]:




from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Abaixo é criado um classificador MLP com 20 neurônios na camada intermediária e dropout de 50%. Foi utilizada a função de ativação sigmoid, a função de loss mean squared error e o otimizador RMSProp. O treinamento é realizado por 200 épocas com batch size de 64.

model = Sequential()
model.add(Dense(20, input_dim=X_train.shape[1], activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

print(model.summary())hist = model.fit(X_train, y_train, batch_size=64, nb_epoch=200, verbose=False)import matplotlib.pyplot as plt
%matplotlib inlinedef plot_line(loss, ylabel, xlabel='Epochs'):
    fig = plt.figure()
    plt.plot(loss)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()
    plt.show()plot_line(hist.history['acc'], 'accuracy')plot_line(hist.history['loss'], 'loss')loss, accuracy = model.evaluate(v_X, v_y)

print("Loss: ", loss)
print("Accuracy: ", accuracy)prefixo_arquivo = 'submissions/titanic-submission'
sufixo_arquivo = '07jul'
name = 'mlptf'

print(model, '\n')

# executar previsão usando o modelo
y_pred = model.predict(X_test).ravel()
vfunc = np.vectorize(lambda x: 'yes' if x > 0.5 else 'no')

# gerar dados de envio (submissão)
submission = pd.DataFrame({
  'person': X_test.index,
  'survived': vfunc(y_pred)
})
submission.set_index('person', inplace=True)

# gerar arquivo CSV para o envio
filename = '%s-p-%s-%s.csv' % (prefixo_arquivo, sufixo_arquivo, name.lower())
submission.to_csv(filename)
# #### LightGBM
# 
# - https://github.com/microsoft/LightGBM
# - https://lightgbm.readthedocs.io/en/latest/
# - https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc
!pip install lightgbmfrom lightgbm import LGBMClassifier
model = LGBMClassifier(random_state=42, objective='binary', metric='binary_logloss',
                      boosting_type='goss', n_estimators=50, max_depth=11)

params = dict(
    boosting_type=['gbdt', 'dart', 'goss'], #rf
    objective=['binary'],
    metric=['binary_logloss'],
    #bagging_freq=[1, 5, 10],
    #bagging_fraction=[0.25, 0.5],
    n_estimators=[50, 75, 100, 200],
    max_depth=[3, 5, 7, 9, 11]
)
#fine_tune_model(model, params, X_train, y_train)

score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)
add_model_info('LGBM', model, score, stddev, elapsed)import lightgbm as lgb

d_train = lgb.Dataset(X_train.values, label=y_train.values)

params = {}
params['random_state'] = 42
params['objective'] = 'binary'
params['boosting_type'] = 'goss' # gbdt, rf, dart, goss

#params['learning_rate'] = 0.003
#params['metric'] = 'binary_logloss'
#params['sub_feature'] = 0.5
#params['num_leaves'] = 10
#params['min_data'] = 50
#params['max_depth'] = 10

#model = lgb.train(params, d_train, 100)
model = lgb.cv(params, d_train, 100, nfold=10)prefixo_arquivo = 'submissions/titanic-submission'
sufixo_arquivo = '08jul'
name = 'lgbm_' + params['boosting_type']

print(model, '\n')

# executar previsão usando o modelo
y_pred = clf.predict(X_test.values)
vfunc = np.vectorize(lambda x: 'yes' if x > 0.5 else 'no')

# gerar dados de envio (submissão)
submission = pd.DataFrame({
  'person': X_test.index,
  'survived': vfunc(y_pred)
})
submission.set_index('person', inplace=True)

# gerar arquivo CSV para o envio
filename = '%s-p-%s-%s.csv' % (prefixo_arquivo, sufixo_arquivo, name.lower())
submission.to_csv(filename)
# ### Ensemble Learning Model
# 
# - https://towardsdatascience.com/automate-stacking-in-python-fc3e7834772e
# - https://github.com/vecxoz/vecstack

# In[51]:


estimators =  [
    ('RF', RandomForestClassifier(random_state=42, n_estimators=100, max_features='auto', max_depth=5)),
    ('BC', BaggingClassifier(random_state=42, n_estimators=100, max_samples=0.25, max_features=0.8)),
    ('GB', GradientBoostingClassifier(random_state=42, max_depth=4, max_features=0.75,
                                   n_estimators=100, learning_rate=0.1, subsample=0.6)),
#    ('XGB', XGBClassifier(max_depth=3, n_estimators=100)),
]
model = VotingClassifier(estimators, n_jobs=-1, weights=(2,1,1))

#estimators, weights=None, n_jobs=None

params = dict(
    weights=[(1,1,1), (2,1,1), (3,1,1), (3,2,1), (2,2,1), (2,1,2), (5,4,3), (1,2,1), (1,1,2), ]
)
#fine_tune_model(model, params, X_train, y_train)

score, stddev, elapsed = evaluate_classification_model(model, X_train, y_train)
add_model_info('VC', model, score, stddev, elapsed)

model = SGDClassifier(random_state=42, max_iter=100, tol=0.1)

params = {'max_iter':[100, 200, 350, 500, 1000], 'tol':[0.1, 0.01]}
model = Perceptron(random_state=42, max_iter=100, tol=0.01)

params = dict(
    max_iter=[100, 200, 350, 500, 1000],
    tol=[0.1, 0.01, 0.001]
)
model = LinearSVC(random_state=42, max_iter=1000, C=10)

params = dict(
    C=[0.001, 0.01, 0.1, 1, 10, 100]
)

# In[ ]:





# ## Avaliar importância dos atributos no modelo

# In[52]:


model = RandomForestClassifier(random_state=42, max_features='auto', n_estimators=100)
model.fit(X_train, y_train)

importances = pd.DataFrame({'feature': X_train.columns,
                            'importance': np.round(model.feature_importances_, 3)})
importances = importances.sort_values('importance', ascending=False).set_index('feature')
importances.head(20)


# In[53]:


importances.plot.bar()


# ## Comparação final entre os algoritmos

# In[54]:


results = pd.DataFrame({'Algorithm': names, 'Score': scores, 'Std Dev': stddevs, 'Time (ms)': times})
results.sort_values(by='Score', ascending=False)


# ## Gerar arquivos com resultados

# In[55]:


# criar diretório para os arquivos de envio
get_ipython().system('test -d submissions || mkdir submissions')


# In[56]:


prefixo_arquivo = 'submissions/titanic-submission'
sufixo_arquivo = '08jul'

for name, model in models:
    print(model, '\n')
    
    # treinar o modelo
    model.fit(X_train, y_train)
    
    # executar previsão usando o modelo
    y_pred = model.predict(X_test)
    vfunc = np.vectorize(lambda x: 'yes' if x > 0 else 'no')

    # gerar dados de envio (submissão)
    submission = pd.DataFrame({
      'person': X_test.index,
      'survived': vfunc(y_pred)
    })
    submission.set_index('person', inplace=True)

    # gerar arquivo CSV para o envio
    filename = '%s-p-%s-%s.csv' % (prefixo_arquivo, sufixo_arquivo, name.lower())
    submission.to_csv(filename)


# In[57]:


get_ipython().system('head submissions/*.csv')


# In[ ]:




