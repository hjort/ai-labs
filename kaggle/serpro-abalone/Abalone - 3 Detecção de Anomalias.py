#!/usr/bin/env python
# coding: utf-8

# ## Importação dos pacotes

# In[1]:


# importar pacotes necessários
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# definir parâmetros extras
import warnings
warnings.filterwarnings("ignore")
sns.set(style="white", color_codes=True)
# ## Carga dos dados de entrada

# In[2]:


# carregar arquivo de dados de treino
filename = 'abalone-train.csv'
#filename = 'https://github.com/hjort/ai-labs/raw/master/kaggle/serpro-abalone/abalone-train.csv'
data = pd.read_csv(filename, index_col='id')

# mostrar alguns exemplos de registros
data.head()


# In[3]:


print(data.shape)


# In[4]:


# selecionar colunas a serem usadas na detecção de anomalias
cols = ['length', 'diameter', 'height', 'whole_weight',
       'shucked_weight', 'viscera_weight', 'shell_weight']
#print(data.columns.values)


# ## Métodos de detecção de outliers

# ### 1. via gráficos de dispersão

# In[5]:


colors = {'M': '#4c72b0', 'F': '#dd8452', 'I': 'green'}

# parâmetros
rows_count = 4
cols_count = 2
col_x = 'rings'

plt.figure(figsize=(16, 32))

i = 0
for col_y in cols:
    plt.subplot(rows_count, cols_count, (i + 1))
    plt.scatter(data[col_x], data[col_y], c=data['sex'].map(colors))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.title(col_x + ' x ' + col_y)
    plt.grid()
    i += 1

plt.show()


# In[6]:


outliers = np.concatenate((
    data[(data['length'] < 0.1) | (data['length'] > 0.75)].index,
    data[(data['length'] < 0.75) & (data['rings'] > 22)].index,
    data[(data['diameter'] < 0.1) | (data['diameter'] > 0.6)].index,
    data[(data['diameter'] < 0.6) & (data['rings'] > 22)].index,
    data[(data['height'] < 0.02) | (data['height'] > 0.3)].index,
    data[(data['height'] < 0.3) & (data['rings'] > 22)].index,
    data[(data['whole_weight'] < 0.05) | (data['whole_weight'] > 2.4)].index,
    data[(data['whole_weight'] < 2.4) & (data['rings'] > 22)].index,
    data[(data['shucked_weight'] < 0.02) | (data['shucked_weight'] > 1.1)].index,
    data[(data['shucked_weight'] < 1.1) & (data['rings'] > 22)].index,
    data[(data['viscera_weight'] < 0.01) | (data['viscera_weight'] > 0.55)].index,
    data[(data['viscera_weight'] < 0.55) & (data['rings'] > 22)].index,
    data[(data['shell_weight'] < 0.02) | (data['shell_weight'] > 0.7)].index,
    data[(data['shell_weight'] < 0.7) & (data['rings'] > 22)].index,
), axis=0)
#print('Número de outliers:', len(outliers))

outliers = np.concatenate((
    data[data.height == 0.0].index,
    data[(data['viscera_weight'] > 0.5) & (data['rings'] < 20 - 1.5)].index,
    data[(data['viscera_weight'] < 0.5) & (data['rings'] > 25 - 1.5)].index,
    data[(data['shell_weight'] > 0.6) & (data['rings'] < 25 - 1.5)].index,
    data[(data['shell_weight'] < 0.8) & (data['rings'] > 25 - 1.5)].index,
    data[(data['shucked_weight'] >= 1.0) & (data['rings'] < 20 - 1.5)].index,
    data[(data['shucked_weight'] < 1.0)  & (data['rings'] > 20 - 1.5)].index,
    data[(data['whole_weight'] >= 2.5) & (data['rings'] < 25 - 1.5)].index,
    data[(data['whole_weight'] < 2.5)  & (data['rings'] > 25 - 1.5)].index,
    data[(data['diameter'] < 0.1)  & (data['rings'] <  5 - 1.5)].index,
    data[(data['diameter'] < 0.6)  & (data['rings'] > 25 - 1.5)].index,
    data[(data['diameter'] >= 0.6) & (data['rings'] < 25 - 1.5)].index,
    data[(data['height'] > 0.4) & (data['rings'] < 15 - 1.5)].index,
    data[(data['height'] < 0.4) & (data['rings'] > 25 - 1.5)].index,
    data[(data['length'] < 0.1)  & (data['rings'] <  5 - 1.5)].index,
    data[(data['length'] < 0.8)  & (data['rings'] > 25 - 1.5)].index,
    data[(data['length'] >= 0.8) & (data['rings'] < 25 - 1.5)].index
), axis=0)
#print('Número de outliers:', len(outliers))
# In[7]:


# marcar registros com anomalia
data['outlier1'] = data.index.map(lambda idx: 1 if idx in outliers else 0)


# In[8]:


print('Número de outliers:', len(data[data.outlier1 == 1].index))

data.loc[outliers].head()data[data.outlier1 == 1].head()
# ### 2. via Z-Score
# 
# - https://medium.com/datadriveninvestor/finding-outliers-in-dataset-using-python-efc3fce6ce32

# In[9]:


threshold = 3

for col in cols:
#col = 'length'
#if True:
    
    z_col = 'z_' + col
    o_col = 'o_' + col

    col_mean = data[col].mean()
    col_stddev = data[col].std()
    #print('Column:  %s\nMean:    %.2f\nStd Dev: %.4f\n' % (col, col_mean, col_stddev))

    data[z_col] = (data[col] - col_mean) / col_stddev
    data[o_col] = data[z_col].apply(lambda z: 1 if np.abs(z) > threshold else 0)

data.head(10).Tdata[data.o_length != 0]
# In[10]:


data['outlier2'] = data['o_length'] + data['o_diameter'] + data['o_height'] +                     data['o_whole_weight'] + data['o_shucked_weight'] +                     data['o_viscera_weight'] + data['o_shell_weight']

# considerar apenas a variável 'height'
#data['outlier2'] = data['o_height']

# marcar registros com anomalia
data['outlier2'] = data['outlier2'].apply(lambda x: 1 if (x > 0) else 0)

data.outlier2 = data.outlier2.astype('category')

#data[data.outlier2 != 0]


# In[11]:


print('Número de outliers:', len(data[data.outlier2 == 1].index))


# ### 3. via IQR
# 
# - https://medium.com/datadriveninvestor/finding-outliers-in-dataset-using-python-efc3fce6ce32

# In[12]:


iqr_factor = 1.5

for col in cols:
#col = 'length'
#if True:

    o_col = 'o_' + col

    col_q1 = data[col].quantile(0.25)
    col_q3 = data[col].quantile(0.75)
    col_iqr = col_q3 - col_q1
    col_lower = col_q1 - (1.5 * col_iqr)
    col_upper = col_q3 + (1.5 * col_iqr)
    
    #print('Column: %s\nQ1:     %.2f\nQ3:     %.4f\nIQR:    %.4f\nLower:  %.4f\nUpper:  %.4f\n' % \
    #      (col, col_q1, col_q3, col_iqr, col_lower, col_upper))

    data[o_col] = data[col].apply(lambda x: 1 if (x < col_lower or x > col_upper) else 0)


# In[13]:


data['outlier3'] = data['o_length'] + data['o_diameter'] + data['o_height'] +                     data['o_whole_weight'] + data['o_shucked_weight'] +                     data['o_viscera_weight'] + data['o_shell_weight']

# considerar apenas a variável 'height'
#data['outlier3'] = data['o_height']

# marcar registros com anomalia
data['outlier3'] = data['outlier3'].apply(lambda x: 1 if (x > 0) else 0)

data.outlier3 = data.outlier3.astype('category')

#data[data.outlier3 != 0]


# In[14]:


print('Número de outliers:', len(data[data.outlier3 == 1].index))


# ### 4-10. via PyOD
# 
# - https://pyod.readthedocs.io/en/latest/
# - https://www.analyticsvidhya.com/blog/2019/02/outlier-detection-python-pyod/

# In[15]:


from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF

data.plot.scatter("height", "length")from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
data['s_height'] = None
data['s_length'] = None
data[['s_height', 's_length']] = scaler.fit_transform(data[['height', 'length']])
data[['height', 'length', 's_height', 's_length']].head()X1 = data['s_height'].values.reshape(-1,1)
X2 = data['s_length'].values.reshape(-1,1)

X = np.concatenate((X1, X2), axis=1)
X[:10]
# In[16]:


outliers_fraction = 0.04 # percentual de "contaminação" (4,0%)
sufixo_arquivo = '%.2f' % (outliers_fraction * 100)

random_state = np.random.RandomState(42)

# modelos a serem usados na detecção de anomalias
classifiers = {
    'Angle-based Outlier Detector (ABOD)': ABOD(contamination=outliers_fraction),
    'Cluster-based Local Outlier Factor (CBLOF)': CBLOF(contamination=outliers_fraction,
                                                        check_estimator=False, random_state=random_state),
    'Feature Bagging': FeatureBagging(LOF(n_neighbors=35), contamination=outliers_fraction,
                       check_estimator=False, random_state=random_state),
    'Histogram-base Outlier Detection (HBOS)': HBOS(contamination=outliers_fraction),
    'Isolation Forest': IForest(contamination=outliers_fraction, random_state=random_state),
    'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
    'Average KNN': KNN(method='mean', contamination=outliers_fraction)
}


# In[17]:


# seleção dos dados a serem usados na detecção automática
X = data[cols]
#X = data[['length', 'height']]
#X = data[['length', 'height', 'whole_weight', 'rings']]
#X.info()


# In[18]:


# executar os algoritmos de detecção automática
outlier_start = 4
for i, (clf_name, clf) in enumerate(classifiers.items()):
    print(i + 1, clf)
    clf.fit(X)
    col = 'outlier' + str(i + outlier_start)
    data[col] = clf.labels_
    data[col] = data[col].astype('category')

data.sample(10)
# ### 11. via Diferença de um Modelo Preditivo

# In[19]:


X = data[cols]
y = data['rings']


# In[20]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import OrthogonalMatchingPursuit

#model = LinearRegression(fit_intercept=False, normalize=True)
model = OrthogonalMatchingPursuit(n_nonzero_coefs=7, fit_intercept=False, normalize=True)

model.fit(X, y)

data['predict'] = model.predict(X)
data['diff'] = abs(data.rings - data.predict)


# In[21]:


data[['rings', 'predict', 'diff']].describe()


# In[22]:


# definir o limiar para ser considerado anomalia
limiar = data['diff'].mean() + 2.25 * data['diff'].std()

data[['rings', 'predict', 'diff']].head()

data[data['diff'] > limiar].head(10)
# In[23]:


# marcar registros com anomalia
data['outlier11'] = data['diff'].apply(lambda x: 1 if (x > limiar) else 0)


# In[24]:


print('Número de outliers:', len(data[data['outlier11'] == 1].index))


# ## Visualização de gráficos
# gerar gráfico em pares com kde nas diagonais
sns.pairplot(data, hue="outlier1", size=3, diag_kind="kde", vars=cols)sns.FacetGrid(data, hue="outlier11", size=4) \
   .map(plt.scatter, "height", "length") \
   .add_legend()# gerar um gráfico do tipo boxplot sobre uma característica individual
plt.figure(figsize=(16, 9))
sns.boxplot(x="rings", y="length", data=data)
# In[25]:


# gerar gráficos de dispersão para cada um dos modelos de outliers

colors = {0: '#4c72b0', 1: '#dd8452'}

# parâmetros
outlier_start = 1
outlier_count = 11
rows_count = 3
cols_count = 4
col_x = 'height'
col_y = 'length'

plt.figure(figsize=(14, 10))

for i in range(outlier_count):
    col = 'outlier' + str(i + outlier_start)
    plt.subplot(rows_count, cols_count, (i + 1))
    plt.scatter(data[col_x], data[col_y], c=data[col].map(colors))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.title(col)

plt.show()


# ## Geração de arquivos sem as anomalias

# In[26]:


get_ipython().system('mkdir input/')


# In[27]:


# gerar arquivos CSV sem as anomalias
outlier_start = 1
outlier_count = 1 #11

pcols = cols + ['sex', 'rings']
for i in range(outlier_start, outlier_count - outlier_start + 1 + 1):
    col = 'outlier' + str(i)
    filename = 'abalone-train-o' + str(i) + '-' + sufixo_arquivo + '.csv'
    print('Gerando arquivo:', filename)
    data[data[col] == 0][pcols].to_csv('input/' + filename)


# In[28]:


get_ipython().system('wc -l  input/abalone-train-o*.csv')


# In[ ]:




