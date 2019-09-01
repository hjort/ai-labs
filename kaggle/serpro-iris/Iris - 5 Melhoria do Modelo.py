#!/usr/bin/env python
# coding: utf-8

# ## Importação dos pacotes

# In[1]:


# importar pacotes necessários
import numpy as np
import pandas as pd

# definir parâmetros extras
#pd.set_option('precision', 2)
pd.set_option('display.max_columns', 100)
# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# importar pacotes usados na seleção do modelo e na medição da precisão
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

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


# ## Carga dos dados de entrada

# In[4]:


# carregar arquivo de dados de treino
data = pd.read_csv('iris-train.csv', index_col='Id')

# mostrar alguns exemplos de registros
data.head()


# In[29]:


# definir dados de entrada

#X = data.drop(['Species'], axis=1) # tudo, exceto a coluna alvo
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = data['Species'] # apenas a coluna alvo

print('Forma dos dados originais:', X.shape, y.shape)


# In[30]:


X.head()


# ## Transformações nos dados
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(y.values)

y = encoder.transform(y)# normalizar os valores numéricos
#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#scaler = MinMaxScaler()
scaler = StandardScaler()
#X_norm = X.copy()
#X_norm[X_norm.columns] = scaler.fit_transform(X_norm[X_norm.columns])
X_norm = pd.DataFrame(scaler.fit_transform(X.values), columns=X.columns, index=X.index)X_norm = (X - np.min(X)) / (np.max(X) - np.min(X))
X_norm.head()# remover apenas um atributo
X_part = X.drop(['PetalLengthCm'], axis=1)from sklearn.decomposition import PCA

pca = PCA(n_components=3)

X_pca = pd.DataFrame(pca.fit_transform(X.values), columns=['pc1','pc2','pc3'], index=X.index)
# ## Separação dos dados de treino e teste

# In[31]:


# separar dados para fins de treino (70%) e de teste (30%)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print('Forma dos dados separados:', X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train.head()

# separar dados normalizados para fins de treino (70%) e de teste (30%)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_norm, y, test_size=0.3, random_state=42)

print('Forma dos dados normalizados separados:', X_train2.shape, X_test2.shape, y_train2.shape, y_test2.shape)

X_train2.head()# separar dados parciais para fins de treino (70%) e de teste (30%)

X_train3, X_test3, y_train3, y_test3 = train_test_split(X_part, y, test_size=0.3, random_state=42)

print('Forma dos dados parciais separados:', X_train3.shape, X_test3.shape, y_train3.shape, y_test3.shape)

X_train3.head()# separar dados em PCA para fins de treino (70%) e de teste (30%)

X_train4, X_test4, y_train4, y_test4 = train_test_split(X_pca, y, test_size=0.3, random_state=42)

print('Forma dos dados em PCA separados:', X_train4.shape, X_test4.shape, y_train4.shape, y_test4.shape)

X_train4.head()
# ## Treinamento dos modelos preditivos

# In[32]:


def evaluate_model_cv(model, X=X, y=y):
  kfold = KFold(n_splits=10, random_state=42)
  results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy', verbose=1)
  score = results.mean() * 100
  stddev = results.std() * 100
  print(model, '\nCross-Validation Score: %.2f (%.2f) %%' % (score, stddev))
  return score

# treina o modelo especificado e avalia a sua respectiva precisão
def fit_and_evaluate(model, X_train, y_train, X_test, y_test):
    
  # treinar o modelo
  model.fit(X_train, y_train)

  # calcular precisão (forma simples)
  score = model.score(X_test, y_test) * 100
  print(model, '\nModel Score: %.2f %%' % score)

  # exibir matriz de confusão
  y_pred = model.predict(X_test)
  cm = confusion_matrix(y_test, y_pred)
  print('Confusion Matrix:\n', cm)# treina e avalia o modelo especificado usando os dados de treino e teste
def evaluate_model_simple(model):
  fit_and_evaluate(model, X_train, y_train, X_test, y_test)

  #print('\nNormalized:')
  #fit_and_evaluate(model, X_train2, y_train2, X_test2, y_test2)
# In[33]:


# faz o ajuste fino do modelo, calculando os melhores hiperparâmetros
def fine_tune_model(model, params, X=X, y=y):
  print('\nFine Tuning Model:')
  print(model, "\nparams:", params)

  kfold = KFold(n_splits=10, random_state=42)
  grid = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=kfold, verbose=1)
  grid.fit(X, y)

  print('\nGrid Score: %.2f %%' % (grid.best_score_ * 100))
  print('Best Params:', grid.best_params_)
 
  return grid

def fine_tune_model_all_data(model, params):
    
  print('\n=> Using Original Data')
  fine_tune_model(model, params, X, y, X_train, y_train)
    
  #print('\n=> Using Scaled Data')
  #fine_tune_model(model, params, X_train2, y_train2)
    
  #print('\n=> Using Partial Data')
  #fine_tune_model(model, params, X_train3, y_train3)
    
  #print('\n=> Using PCA-Decomposed Data')
  #fine_tune_model(model, params, X_train4, y_train4)
# # Avaliação e ajuste fino de cada modelo preditivo

# In[34]:


# A) Logistic Regression
model = LogisticRegression(random_state=42, solver='lbfgs', multi_class='auto', max_iter=500, C=10)
evaluate_model_cv(model)

params = {'solver':['liblinear', 'lbfgs'], 'C':np.logspace(-3,3,7)}
#fine_tune_model(model, params)


# In[35]:


# B) Decision Tree
model = DecisionTreeClassifier(random_state=42, max_depth=5, criterion='entropy')
evaluate_model_cv(model)

params = {'criterion':['gini','entropy'], 'max_depth':[3,5,7,11]}
#fine_tune_model(model, params)


# In[36]:


# C) K-Nearest Neighbours
model = KNeighborsClassifier(n_neighbors=3)
evaluate_model_cv(model)

params = {'n_neighbors':[1, 3, 5, 7, 9]}
#fine_tune_model(model, params)


# In[37]:


# D) Support Vector Machine (SVM)
model = SVC(random_state=42, C=1, gamma=0.001, kernel='linear')
evaluate_model_cv(model)

params = {'C':[0.001, 0.01, 0.1, 1, 10, 100], 'gamma':[0.001, 0.01, 0.1, 1, 10, 100], 'kernel':['linear', 'rbf']}
#fine_tune_model(model, params)


# In[38]:


# E) Random Forest
model = RandomForestClassifier(random_state=42, max_features='auto', n_estimators=100)
evaluate_model_cv(model)

params = {'n_estimators':[10, 50, 100, 500], 'max_features':['auto', 'sqrt', 'log2']}
#fine_tune_model(model, params)


# In[39]:


# F) Stochastic Gradient Descent (SGD)
model = SGDClassifier(random_state=42, max_iter=100, tol=0.01)
evaluate_model_cv(model)

params = {'max_iter':[100, 200, 350, 500, 1000], 'tol':[0.1, 0.01]}
#fine_tune_model(model, params)


# In[16]:


# G) Perceptron
model = Perceptron(random_state=42, max_iter=100, tol=0.01)
evaluate_model_cv(model)

params = {'max_iter':[100, 200, 350, 500, 1000], 'tol':[0.1, 0.01, 0.001]}
#fine_tune_model(model, params)


# In[17]:


# H) Naïve Bayes
model = GaussianNB(priors=None, var_smoothing=1e-08)
evaluate_model_cv(model)

params = {'priors': [None], 'var_smoothing': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]}
#fine_tune_model(model, params)


# In[18]:


# I) Linear SVM
model = LinearSVC(random_state=42, max_iter=1000, C=1)
evaluate_model_cv(model)

params = {'C':[0.001, 0.01, 0.1, 1, 10, 100]}
#fine_tune_model(model, params)


# In[19]:


# J) Ada Boost
model = AdaBoostClassifier(DecisionTreeClassifier(random_state=42), n_estimators=1)
evaluate_model_cv(model)

params = {'n_estimators':[1,3,5,7,11]}
#fine_tune_model(model, params)


# In[20]:


# K) Gradient Boosting
model = GradientBoostingClassifier(random_state=42, max_depth=5)
evaluate_model_cv(model)

'''
params = {
    "learning_rate":[0.01, 0.05, 0.1],
    "max_depth":[3, 5, 7],
    "max_features":["log2", "sqrt"],
    "criterion":["friedman_mse", "mae"],
    "subsample":[0.5, 0.75, 1.0],
}
'''

params = {'max_depth':[3, 5, 7]}
#fine_tune_model(model, params)

# L) Ridge
model = Ridge(random_state=42, alpha=1)
evaluate_model_cv(model)

#params = {'alpha':[1,0.1,0.01,0.001,0.0001,0]}
#fine_tune_model(model, params)
# In[21]:


# M) Multi-Layer Perceptron (MLP)
model = MLPClassifier(random_state=42, solver='lbfgs', alpha=1, hidden_layer_sizes=(15,))
evaluate_model_cv(model)

params = {'alpha':[1,0.1,0.01,0.001,0.0001,0]}
#fine_tune_model(model, params)


# In[22]:


# N) Linear Discriminant Analysis (LDA)
model = LinearDiscriminantAnalysis(solver='svd')
evaluate_model_cv(model)

params = {'solver':['svd', 'lsqr', 'eigen']}
#fine_tune_model(model, params)


# ## Comparação final entre os algoritmos

# In[23]:


models = []
models.append(('LR', LogisticRegression(random_state=42, solver='lbfgs', multi_class='auto', max_iter=500, C=10)))
models.append(('DT', DecisionTreeClassifier(random_state=42, max_depth=5, criterion='entropy')))
models.append(('KNN', KNeighborsClassifier(n_neighbors=3)))
models.append(('SVM', SVC(random_state=42, C=1, gamma=0.001, kernel='linear')))
models.append(('RF', RandomForestClassifier(random_state=42, max_features='auto', n_estimators=100)))
models.append(('SGD', SGDClassifier(random_state=42, max_iter=100, tol=0.01)))
models.append(('NN', Perceptron(random_state=42, max_iter=100, tol=0.01)))
models.append(('NB', GaussianNB(priors=None, var_smoothing=1e-08)))
models.append(('LSVM', LinearSVC(random_state=42, max_iter=1000, C=1)))
models.append(('ABDT', AdaBoostClassifier(DecisionTreeClassifier(random_state=42), n_estimators=1)))
models.append(('GB', GradientBoostingClassifier(random_state=42, max_depth=5)))
models.append(('MLP', MLPClassifier(random_state=42, solver='lbfgs', alpha=1, hidden_layer_sizes=(15,))))
models.append(('LDA', LinearDiscriminantAnalysis(solver='svd')))


# In[24]:


results = []
names = []
scores = []
stddevs = []

for name, model in models:
  kfold = KFold(n_splits=10, random_state=42)
  cv_results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
  results.append(cv_results)
  names.append(name)
  scores.append(cv_results.mean() * 100)
  stddevs.append(cv_results.std() * 100)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)


# In[25]:


# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[26]:


results = pd.DataFrame({'Model': names, 'Score': scores, 'Std Dev': stddevs})
results.sort_values(by='Score', ascending=False)


# ## Verificação contra os dados reais (resultados esperados)

# In[27]:


# carregar arquivo de dados de teste
test_data = pd.read_csv('iris-test.csv', index_col='Id')

# mostrar alguns exemplos de registros
test_data.head()


# In[ ]:


# carregar arquivo de dados de teste
real_data = pd.read_csv('iris-solution.csv', index_col='Id')

# mostrar alguns exemplos de registros
real_data.head()


# In[92]:


names = []
scores = []
for name, model in models:
  X_train, X_test, y_train, y_test = X, test_data, y, real_data

  # treinar o modelo
  model.fit(X_train, y_train)

  # calcular precisão (forma simples)
  score = model.score(X_test, y_test) * 100
  print(model, '\nModel Score: %.2f %%' % score)

  # exibir matriz de confusão
  y_pred = model.predict(X_test)
  cm = confusion_matrix(y_test, y_pred)
  print('Confusion Matrix:\n', cm, '\n')
    
  names.append(name)
  scores.append(score)


# In[91]:


results = pd.DataFrame({'Model': names, 'Score': scores})
results.sort_values(by='Score', ascending=False)


# In[ ]:




