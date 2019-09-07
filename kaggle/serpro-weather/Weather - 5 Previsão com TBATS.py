#!/usr/bin/env python
# coding: utf-8

# # SERPRO - Weather
# 
# _Previsão de temperaturas médias com base em dados históricos_
# 
# https://www.kaggle.com/c/serpro-weather/

# ## Referências
# 
# ### Forecasting Time Series with Multiple Seasonalities using TBATS in Python
# - https://medium.com/intive-developers/forecasting-time-series-with-multiple-seasonalities-using-tbats-in-python-398a00ac0e8a
# 
# ### Forecasting Time Series With Complex Seasonal Patterns Using Exponential Smoothing
# - https://www.tandfonline.com/doi/abs/10.1198/jasa.2011.tm09771
# 
# ### TBATS Model (Exponential Smoothing State Space Model With Box-Cox Transformation, ARMA Errors, Trend And Seasonal Components)
# - https://www.rdocumentation.org/packages/forecast/versions/8.4/topics/tbats

# ## Importação dos pacotes

# In[1]:


# importar pacotes necessários
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#import matplotlib.pyplot as plt
import pickle

# instalar pacotes especiais
!pip install tbats
# In[2]:


# definir parâmetros extras
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6


# ## Definição de parâmetros

# In[3]:


# definir períodos a serem considerados no modelo
periodo1 = 1.0
periodo2 = 365.25

# definir data inicial de corte
data_inicio_amostra = '2014-11'

## 4.82930 4.62694
periodo1 = 1.0; periodo2 = 365.25; data_inicio_amostra = '2014-11'

## 4.89313 4.72361
periodo1 = 7.0; periodo2 = 365.25; data_inicio_amostra = '2014-11'
# In[4]:


print('Períodos sazonais: (%.2f, %.2f)' % (periodo1, periodo2))
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
##data = data.resample(frequencia).mean()

# filtrar período desejado
#data = data[data_inicio_amostra:]
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
#ts = ts.asfreq('D')
ts.head()


# In[12]:


# plotar a série temporal
plt.plot(ts)
plt.title('Temperatura ao longo dos anos (em graus Celsius)', fontsize=20)
plt.show()


# In[13]:


# https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html#visualization-lag

from pandas.plotting import lag_plot

lag_plot(ts)


# In[14]:


# https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html#visualization-autocorrelation

# analisar sazonalidade na série através da autocorrelação
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(ts)


# In[15]:


# https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(ts, model='additive', freq=365)
#result = seasonal_decompose(ts + 273.15, model='multiplicative', freq=365)
result.plot()
plt.show()


# ## Modelagem preditiva

# ### Preditor automático TBATS
# - https://github.com/intive-DataScience/tbats

# In[16]:


# dividir dados entre treino e teste
corte = (len(ts) - 365) # 1 ano
#corte = int(len(ts) * 0.75)
treino = ts[:corte]
teste = ts[corte:]
print('Treino:', treino.shape)
print('Teste: ', teste.shape)


# In[17]:


from tbats import TBATS, BATS

model = TBATS(seasonal_periods=(periodo1, periodo2), use_trend=True)
get_ipython().run_line_magic('time', 'smodel = model.fit(treino)')

print('\n' + smodel.summary())


# ### Avaliação do resultado em previsões futuras

# In[18]:


# realizar a previsão
fitted, confidence = smodel.forecast(steps=len(teste), confidence_level=0.95)
index_of_fc = pd.date_range(teste.index[0], periods=len(teste), freq='D')

# criar séries para plotagem
fitted_series = pd.Series(confidence['mean'], index=index_of_fc)
lower_series = pd.Series(confidence['lower_bound'], index=index_of_fc)
upper_series = pd.Series(confidence['upper_bound'], index=index_of_fc)

# plotar gráfico
plt.plot(ts)
plt.plot(fitted_series, color='darkgreen')
plt.fill_between(lower_series.index,
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)
plt.title("Previsão de temperaturas com TBATS", fontsize=20)
plt.show()


# In[19]:


def rmse(predictions, targets):
    assert len(predictions) == len(targets)
    return np.sqrt(np.mean((predictions - targets) ** 2))

def rmsle(predictions, targets):
    assert len(predictions) == len(targets)
    return np.sqrt(np.mean((np.log(1 + predictions) - np.log(1 + targets)) ** 2))


# In[20]:


print('RMSE:', rmse(fitted_series, teste))


# ### Previsão de valores futuros com os dados de teste

# In[21]:


from tbats import TBATS, BATS

model = TBATS(seasonal_periods=(periodo1, periodo2), use_trend=True)
get_ipython().run_line_magic('time', 'smodel = model.fit(ts)')

print('\n' + smodel.summary())


# In[22]:


# obter intervalo de datas para a previsão
primeira_data = test_data.index[0]
ultima_data = test_data.index[-1]
print('Intervalo de datas: %s => %s' % (primeira_data, ultima_data))
datas_previsao = test_data.index
#datas_previsao


# In[23]:


# gerar novos valores a partir do modelo
periodos_previsao = len(datas_previsao)
valores_previstos, confianca = smodel.forecast(steps=len(datas_previsao), confidence_level=0.95)


# In[24]:


# construir dataframe com previsão
df = pd.DataFrame({
    'date': datas_previsao,
    'temperature': valores_previstos
})
df.set_index('date', inplace=True)
df.info()
df.head()


# In[25]:


df.info()


# In[26]:


# obter intervalo de datas necessário
data_inicio = test_data.index[0]
data_final = test_data.index[-1]
print('Intervalo necessário:', data_inicio, '=>', data_final)


# In[27]:


# restringir ao intervalo de datas esperado
pred_data = df
#pred_data = pred_data[data_inicio:data_final]

# converter temperaturas novamente para graus Celsius
#pred_data['temperature'] -= 273.15

# exibir informações do dataframe
pred_data.info()
pred_data.head()


# ## Geração do arquivo de resultados

# In[28]:


# criar diretório de submissões
get_ipython().system('dd="submissions/"; if [ ! -d $dd ]; then mkdir $dd; fi')


# In[29]:


# gravar arquivo CSV com os resultados
nome_arquivo = 'submissions/weather-submission-tbats-' +     '%.2f-%.2f-' % (periodo1, periodo2) +     data_inicio_amostra + '.csv'
pred_data.to_csv(nome_arquivo)
print('Arquivo gravado com sucesso:', nome_arquivo)

# persistir em arquivo Pickle o modelo preditivo
nome_arquivo = 'weather-model-' + frequencia + '.pickle'
pickle.dump(smodel, open(nome_arquivo, 'wb'))
print('Arquivo gravado com sucesso:', nome_arquivo)
# ## Comparação da previsão com dados reais
# carregar dados previstos
arq = '/home/rodrigo/Downloads/weather-submission-tbats-1.00-365.25-2013-01.csv'
pred_data = pd.read_csv(arq, index_col='date', parse_dates=['date'])
pred_data.head().T
# In[30]:


# carregar dados reais
real_data = pd.read_csv(prefixo_arquivos + 'weather-solution.csv', index_col='date', parse_dates=['date'])
real_data.head().T


# In[ ]:


# comparação com os dados reais
plt.plot(pred_data)
plt.plot(real_data)
plt.title('Temperaturas previstas x reais', fontsize=20)
plt.show()


# ### Medição do Erro Médio Quadrático (RMSE)

# In[ ]:


print('RMSE:', rmse(pred_data['temperature'], real_data['temperature']))


# In[ ]:




