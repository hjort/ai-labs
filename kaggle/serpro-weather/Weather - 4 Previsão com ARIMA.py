#!/usr/bin/env python
# coding: utf-8

# # SERPRO - Weather
# 
# _Previsão de temperaturas médias com base em dados históricos_
# 
# https://www.kaggle.com/c/serpro-weather/

# ## Referências
# 
# ### pmdarima: ARIMA estimators for Python
# - https://www.alkaline-ml.com/pmdarima/
# 
# ### ARIMA Model – Complete Guide to Time Series Forecasting in Python
# - https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
# 
# ### How to Create an ARIMA Model for Time Series Forecasting in Python
# - https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
# 
# ### Time Series Forecasting - ARIMA models - Towards Data Science
# - https://towardsdatascience.com/time-series-forecasting-arima-models-7f221e9eee06
# 
# ### Complete guide to Time Series Forecasting (with Codes in Python)
# - https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/

# ## Importação dos pacotes

# In[1]:


# importar pacotes necessários
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#import matplotlib.pyplot as plt
import pickle

# instalar pacotes especiais
!pip install pmdarima
# In[2]:


# definir parâmetros extras
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6


# ## Definição de parâmetros

# In[3]:


# definir frequência a ser considerada no modelo
frequencia = '2W' # 7D 15D 2W 3W M
periodos_ano = 24

# definir data inicial de corte
data_inicio_amostra = '2013-01'


# In[4]:


# calcular períodos que cabem em um ano
#intervalo_ano = pd.date_range(start='2018-01-01', end='2019-01-01', freq=frequencia)
#periodos_ano = len(intervalo_ano)
print('Frequência:', frequencia)
print('Períodos em um ano:', periodos_ano)
print('Data de início da amostra:', data_inicio_amostra)


# ## Carga dos dados

# In[5]:


prefixo_arquivos = ''
#prefixo_arquivos = 'https://github.com/hjort/ai-labs/raw/master/kaggle/serpro-weather/'


# In[6]:


# carregar arquivo de dados de treino
train_data = pd.read_csv(prefixo_arquivos + 'weather-train.csv', index_col='date', parse_dates=['date'])
train_data.info()
train_data.head()


# In[7]:


# carregar arquivo de dados de teste
test_data = pd.read_csv(prefixo_arquivos + 'weather-test.csv', index_col='date', parse_dates=['date'])
test_data.info()
test_data.head()


# ## Transformações nos dados

# In[8]:


# remover valores nulos
data = train_data.dropna()

# reduzir a quantidade de dados para a frequência
data = data.resample(frequencia).mean()

# filtrar período desejado
data = data[data_inicio_amostra:]
#data = data['2013-01':]
#data = data['2013-01':'2015-12']

# converter temperatura para Kelvin
#data['temperature'] += 273.15

data.head()


# In[9]:


data.info()


# In[10]:


data.describe()


# ## Análise da série temporal

# In[11]:


# criar série temporal a partir do dataframe
ts = data['temperature']
ts = ts.asfreq(frequencia)
ts.head()


# In[12]:


# plotar a série temporal
plt.plot(ts)
plt.title('Temperatura ao longo dos anos (em graus Celsius)', fontsize=20)
plt.show()


# ## Analisar estacionariedade

# In[13]:


from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries, window):
    
    # Determing rolling statistics
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()

    # Plot rolling statistics
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation', fontsize=20)
    plt.show(block=False)
    
    # Perform Dickey-Fuller test
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries)
    dfoutput = pd.Series(dftest[0:4], index=[
        'Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


# In[14]:


# avaliar se a série é estacionária
test_stationarity(ts, periodos_ano)

from statsmodels.tsa.stattools import adfuller
result = adfuller(ts.values)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
# ### Analisar componente de sazonalidade

# In[15]:


# Plot
fig, axes = plt.subplots(2, 1, figsize=(14,8), dpi=100, sharex=True)

# Usual Differencing
axes[0].plot(ts, label='Original Series')
axes[0].plot(ts.diff(1), label='Usual Differencing')
axes[0].set_title('Usual Differencing')
axes[0].legend(loc='upper left', fontsize=10)

# Seasonal 1st
axes[1].plot(ts, label='Original Series')
axes[1].plot(ts.diff(periodos_ano), label='Seasonal Differencing', color='green')
axes[1].set_title('Seasonal Differencing')
plt.legend(loc='upper left', fontsize=10)
plt.suptitle('Temperaturas', fontsize=16)
plt.show()


# ### Decomposição das componentes da série

# In[39]:


from statsmodels.tsa.seasonal import seasonal_decompose

ts_log = ts
decomposition = seasonal_decompose(ts_log, freq=periodos_ano)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Tendência')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Sazonalidade')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Resíduos')
plt.legend(loc='best')
plt.tight_layout()


# ## Modelagem preditiva

# ### Testes para estimar parâmetros do ARIMA

# https://www.alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ndiffs.html

# In[16]:


from pmdarima.arima.utils import ndiffs

dft = pd.DataFrame({
    'Teste': [
        'ADF (Augmented Dickey-Fuller)',
        'KPSS (Kwiatkowski–Phillips–Schmidt–Shin)',
        'PP (Phillips–Perron)'
    ],
    'Valor estimado para o termo "d"': [
        ndiffs(ts, test='adf'),
        ndiffs(ts, test='kpss'),
        ndiffs(ts, test='pp')
    ]
})
dft.set_index('Teste', inplace=True)
dft


# https://www.alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.nsdiffs.html

# In[17]:


from pmdarima.arima.utils import nsdiffs

dft = pd.DataFrame({
    'Teste': [
        'OCSB (Osborn-Chui-Smith-Birchenhall)',
        'CH (Canova-Hansen)'
    ],
    'Valor estimado para o termo "D"': [
        nsdiffs(ts, periodos_ano, test='ocsb'),
        nsdiffs(ts, periodos_ano, test='ch')
    ]
})
dft.set_index('Teste', inplace=True)
dft


# ### Preditor automático ARIMA
# - https://www.alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html

# In[18]:


from pmdarima import auto_arima

smodel = auto_arima(ts, start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=periodos_ano,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)

smodel.summary()

# melhor configuração: SARIMAX(3, 1, 1)x(2, 1, 0, 26)
# https://www.alkaline-ml.com/pmdarima/

from pmdarima.arima import ARIMA

smodel = ARIMA(order=[2, 1, 0], seasonal_order=[2, 1, 0, 12])
smodel.fit(ts)

smodel.summary()# dividir dados entre treino e teste
corte = int(len(ts) * 0.7)
treino = ts[:corte]
teste = ts[corte:]
print('Treino:', treino.shape)
print('Teste: ', teste.shape)treino = ts
# ### Avaliação do resultado em previsões futuras

# In[19]:


# realizar a previsão
fitted, confint = smodel.predict(n_periods=periodos_ano, return_conf_int=True)
index_of_fc = pd.date_range(data.index[-1], periods=periodos_ano, freq=frequencia)

# criar séries para plotagem
fitted_series = pd.Series(fitted, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# plotar gráfico
plt.plot(ts)
plt.plot(fitted_series, color='darkgreen')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)
plt.title("Previsão de temperaturas com SARIMA", fontsize=20)
plt.show()


# ### Previsão de valores futuros com os dados de teste

# In[20]:


# obter intervalo de datas para a previsão
primeira_data = ts.index[-1] + 1
ultima_data = pd.Timestamp(test_data.index[-1], freq='M') + 2
print('Intervalo de datas: %s => %s' % (primeira_data, ultima_data))
datas_previsao = pd.date_range(start=primeira_data, end=ultima_data, freq=frequencia)
datas_previsao


# In[21]:


# gerar novos valores a partir do modelo
periodos_previsao = len(datas_previsao)
valores_previstos = smodel.predict(n_periods=periodos_previsao)


# In[22]:


# construir dataframe com previsão
df = pd.DataFrame({
    'date': datas_previsao,
    'temperature': valores_previstos
})
df.set_index('date', inplace=True)
df.info()
df.head()

data.iloc[-2:]
# In[23]:


# inserir na primeira posição os últimos valores de treino
df = (data.iloc[-2:]).append(df)
df.info()


# In[24]:


# obter intervalo de datas necessário
data_inicio = test_data.index[0]
data_final = test_data.index[-1]
print('Intervalo necessário:', data_inicio, '=>', data_final)


# In[25]:


# interpolar dados para obter valores diários
pred_data = df.resample('D').interpolate(method='cubic')

# restringir ao intervalo de datas esperado
pred_data = pred_data[data_inicio:data_final]

# converter temperaturas novamente para graus Celsius
#pred_data['temperature'] -= 273.15

# exibir informações do dataframe
pred_data.info()
pred_data.head()


# ## Geração do arquivo de resultados

# In[26]:


# criar diretório de submissões
get_ipython().system('dd="submissions/"; if [ ! -d $dd ]; then mkdir $dd; fi')


# In[27]:


# gravar arquivo CSV com os resultados
nome_arquivo = 'submissions/weather-submission-arima-' +     frequencia + 'x' + str(periodos_ano) + '-' + data_inicio_amostra + '.csv'
pred_data.to_csv(nome_arquivo)
print('Arquivo gravado com sucesso:', nome_arquivo)

# persistir em arquivo Pickle o modelo preditivo
nome_arquivo = 'weather-model-' + frequencia + '.pickle'
pickle.dump(smodel, open(nome_arquivo, 'wb'))
print('Arquivo gravado com sucesso:', nome_arquivo)
# ## Comparação da previsão com dados reais

# In[28]:


# carregar dados reais
real_data = pd.read_csv(prefixo_arquivos + 'weather-solution.csv', index_col='date', parse_dates=['date'])
real_data.head()


# In[29]:


# comparação com os dados reais
plt.plot(pred_data)
plt.plot(real_data)
plt.title('Temperaturas previstas x reais', fontsize=20)
plt.show()


# ### Medição do Erro Médio Quadrático (RMSE)

# In[30]:


def rmse(predictions, targets):
    assert len(predictions) == len(targets)
    return np.sqrt(np.mean((predictions - targets) ** 2))

def rmsle(predictions, targets):
    assert len(predictions) == len(targets)
    return np.sqrt(np.mean((np.log(1 + predictions) - np.log(1 + targets)) ** 2))


# In[31]:


print('RMSE:', rmse(pred_data['temperature'], real_data['temperature']))


# In[ ]:




