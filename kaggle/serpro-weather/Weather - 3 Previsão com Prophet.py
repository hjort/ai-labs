#!/usr/bin/env python
# coding: utf-8

# # SERPRO - Weather
# 
# _Previsão de temperaturas médias com base em dados históricos_
# 
# https://www.kaggle.com/c/serpro-weather/

# ## Referências
# 
# ### Prophet Quick Start Guide 
# - https://facebook.github.io/prophet/docs/quick_start.html
# 
# ### GitHub - facebook/prophet
# - https://github.com/facebook/prophet
# 
# ### Analysis of Stock Market Cycles with fbprophet package in Python
# - https://towardsdatascience.com/analysis-of-stock-market-cycles-with-fbprophet-package-in-python-7c36db32ecd0

# ## Importação dos pacotes

# In[2]:


# importar pacotes necessários
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# instalar pacotes especiais
!conda install pystan -y
!pip install fbprophet
# In[3]:


from fbprophet import Prophet


# ## Carga dos dados

# In[4]:


prefixo_arquivos = ''
#prefixo_arquivos = 'https://github.com/hjort/ai-labs/raw/master/kaggle/serpro-weather/'


# In[5]:


# carregar arquivo de dados de treino
train_data = pd.read_csv(prefixo_arquivos + 'weather-train.csv', index_col='date', parse_dates=['date'])
train_data.info()
train_data.head()


# In[9]:


# carregar arquivo de dados de teste
test_data = pd.read_csv(prefixo_arquivos + 'weather-test.csv', index_col='date', parse_dates=['date'])
test_data.info()
test_data.head()


# ## Transformações nos dados

# In[10]:


# ajustar dados de treino para o formato do Prophet
data2 = train_data[['temperature']]
data2 = data2.reset_index()
data2.columns = ['ds', 'y']
data2.head()


# In[11]:


data2.tail()


# ## Modelagem preditiva

# ### Teste do modelo

# In[12]:


# criar e treinar o modelo
model = Prophet(daily_seasonality=False)
model.fit(data2)


# In[16]:


# criar série com dados futuros (2 anos)
future = model.make_future_dataframe(periods=365*2)
future.tail()


# In[17]:


# realizar previsão com dados futuros
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[18]:


# plotar dados reais e previsão
fig1 = model.plot(forecast)


# In[19]:


# decompor tendência e sazonalidades do modelo
fig2 = model.plot_components(forecast)


# ### Avaliar a precisão do modelo

# In[40]:


# dividir os dados em 80% + 20%
divisao = int(data2.shape[0] * 4 / 5)
data2a = data2[:divisao]
data2b = data2[divisao:]
print(data2.shape, '=', data2a.shape, '+', data2b.shape)


# In[41]:


data2a.info()
data2a.head()


# In[42]:


# criar e treinar o modelo
model = Prophet(daily_seasonality=False)
model.fit(data2a)


# In[43]:


# preparar dados futuros
future = data2b.drop(['y'], axis=1)
future.info()
future.head()


# In[44]:


# realizar a previsão
forecast = model.predict(future)
forecast[['ds', 'yhat']].tail()


# In[50]:


# mesclar os dois dataframes novamente
data3 = data2b.merge(forecast)[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']]
data3['diff'] = abs(data3['y'] - data3['yhat'])
data3.info()
data3.head()


# In[51]:


# plotar gráfico comparando valores reais e previstos
plt.figure(figsize=(16, 9))

data3['y'].plot(alpha=0.5, style='-')
data3['yhat'].plot(style=':')
data3['yhat_lower'].plot(style='--')
data3['yhat_upper'].plot(style='--')

plt.legend(['real', 'previsto', 'pmenor', 'pmaior'], loc='upper left')


# ### Medição do Erro Médio Quadrático (RMSE)

# In[52]:


def rmse(predictions, targets):
    assert len(predictions) == len(targets)
    return np.sqrt(np.mean((predictions - targets) ** 2))

def rmsle(predictions, targets):
    assert len(predictions) == len(targets)
    return np.sqrt(np.mean((np.log(1 + predictions) - np.log(1 + targets)) ** 2))


# In[53]:


print('RMSE:', rmse(data3['yhat'], data3['y']))


# ## Previsão de valores futuros com os dados de teste
# https://towardsdatascience.com/basic-time-series-manipulation-with-pandas-4432afee64ea
test_dates = pd.date_range(start='2016-11-17', end='2017-11-29', freq='D')
test_dates
# In[28]:


# criar dados futuros a partir dos dados de teste
future_data = pd.DataFrame(test_data.index.values, columns=['ds'])
future_data.info()
future_data.head()


# In[29]:


# realizar a previsão
forecast = model.predict(future_data)
forecast[['ds', 'yhat']].tail()


# In[31]:


# construir dataframe com previsão
pred_data = pd.DataFrame({
    'date': forecast['ds'],
    'temperature': forecast['yhat']
})
pred_data.set_index('date', inplace=True)
pred_data.info()
pred_data.head()


# ## Geração do arquivo de resultados

# In[32]:


# criar diretório de submissões
get_ipython().system('dd="submissions/"; if [ ! -d $dd ]; then mkdir $dd; fi')


# In[33]:


# gravar arquivo CSV com os resultados
nome_arquivo = 'submissions/weather-submission-prophet.csv'
pred_data.to_csv(nome_arquivo)
print('Arquivo gravado com sucesso:', nome_arquivo)


# ## Comparação da previsão com dados reais

# In[54]:


# carregar dados reais
real_data = pd.read_csv(prefixo_arquivos + 'weather-solution.csv', index_col='date', parse_dates=['date'])
real_data.head()


# In[58]:


# comparação com os dados reais
plt.figure(figsize=(16, 9))
plt.plot(pred_data)
plt.plot(real_data)
plt.title('Temperaturas previstas x reais', fontsize=20)
plt.show()


# In[ ]:




