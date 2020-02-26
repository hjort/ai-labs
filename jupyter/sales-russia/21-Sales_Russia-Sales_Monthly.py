#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importar pacotes necessários
import numpy as np
import pandas as pd


# In[2]:


# definir parâmetros extras
pd.set_option('precision', 4)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


# In[3]:


input_prefix = 'data/'
#input_prefix = 'https://github.com/hjort/ai-labs/raw/master/jupyter/sales-russia/data/'

bzfile = '.bz2' # ''


# In[4]:


#dateparse = lambda x: pd.datetime.strptime(x, '%d.%m.%Y')
df_train = pd.read_csv(    input_prefix + 'sales_train.csv' + bzfile,
    #nrows=300, # somente primeiras linhas!
    #parse_dates=['date'], date_parser=dateparse,
    dtype={'date_block_num': np.int8, 'shop_id': np.int8, 'item_id': np.int16, \
           'item_price': np.float32, 'item_cnt_day': np.int32},
    )
print('shape:', df_train.shape)
df_train.head()


# In[5]:


get_ipython().run_line_magic('time', "df_train['item_price'] = (np.ceil(df_train['item_price'])).astype(np.int32)")


# In[6]:


df_train.info()


# In[7]:


df_train.describe()


# In[8]:


df_train.head()


# In[9]:


df_train2 = df_train.query('item_cnt_day > 0 and item_price > 0')
df_train2.info()


# In[10]:


item_price_monthly_mean = np.ceil(
        df_train2.groupby(
            ['shop_id', 'item_id', 'date_block_num']
        )['item_price'].mean()
    ).astype(np.int32)
item_price_monthly_mean


# In[11]:


item_cnt_monthly_sum = df_train2.groupby(
        ['shop_id', 'item_id', 'date_block_num']
    )['item_cnt_day'].sum()
item_cnt_monthly_sum


# In[12]:


type(item_price_monthly_mean)


# In[13]:


type(item_cnt_monthly_sum)


# In[14]:


item_monthly = pd.concat([item_price_monthly_mean, item_cnt_monthly_sum], axis=1)

#item_monthly = pd.merge(
#    item_price_monthly_mean, item_cnt_monthly_sum, 
#    how='inner', left_index=True, right_index=True)

item_monthly.head()


# In[15]:


del(item_price_monthly_mean)
del(item_cnt_monthly_sum)


# In[16]:


item_monthly.rename(columns={
        'item_price': 'item_price_mean', 'item_cnt_day': 'item_cnt_sum'
    }, inplace=True)
item_monthly.head()


# In[17]:


item_monthly.info()


# In[18]:


item_monthly.describe().T


# In[19]:


item_monthly.index.names


# In[20]:


item_monthly.to_csv('sales_monthly.csv')


# In[21]:


get_ipython().system('head sales_monthly.csv')


# In[22]:


get_ipython().system('rm -f sales_monthly.csv.bz2 && bzip2 -9 sales_monthly.csv')

