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


# ## Sales (testing)

# In[4]:


df_test = pd.read_csv(
    input_prefix + 'test.csv' + bzfile, #nrows=10000,
    dtype={'shop_id': np.int8, 'item_id': np.int16},
    index_col='ID')
print('shape:', df_test.shape)
df_test.head()


# In[5]:


df_test.info()


# In[6]:


df_test.describe()


# In[7]:


# criar dataframe sem quantidade de itens
df_zeroed = df_test.copy()
df_zeroed['key'] = 0
df_zeroed = pd.merge(df_zeroed,
                     pd.DataFrame({'date_block_num': np.arange(0, 34), 'key': np.zeros(34, dtype=int)}),
                     how='left', on='key').\
                set_index(['shop_id', 'item_id', 'date_block_num']).\
                drop(['key'], axis=1)
df_zeroed.head()


# In[8]:


df_zeroed.info()


# In[9]:


df_zeroed.index.names


# ## Sales (monthly)

# In[10]:


df_monthly = pd.read_csv(    input_prefix + '../sales_monthly.csv' + bzfile,
    #nrows=300, # somente primeiras linhas!
    dtype={'date_block_num': np.int8, 'shop_id': np.int8, 'item_id': np.int16,
           'item_price_mean': np.int32, 'item_cnt_sum': np.int32},
    index_col=['shop_id', 'item_id', 'date_block_num']
)
print('shape:', df_monthly.shape)
df_monthly.head()


# In[11]:


df_monthly.info()


# In[12]:


df_monthly.describe()


# In[13]:


df_monthly.index.names


# ## Zeroed joined Monthly Sales

# In[14]:


# mesclar dataframes e zerar valores nulos
df_train2 = pd.merge(df_zeroed,
                     df_monthly,
                     how='left', left_index=True, right_index=True).\
                fillna(0)
df_train2.head()


# In[15]:


df_train2['item_price_mean'] = df_train2['item_price_mean'].astype(np.int32)
df_train2['item_cnt_sum'] = df_train2['item_cnt_sum'].astype(np.int32)


# In[16]:


df_train2.head(10)


# In[17]:


df_train2.info()


# In[18]:


df_train2.describe()


# In[19]:


df_train2.index.names


# ## Shops

# In[20]:


df_shops = pd.read_csv(
    input_prefix + '../shops_full' + '.csv' + bzfile,
    dtype={'city_code': 'category'},
    index_col='shop_id')
print('shape:', df_shops.shape)
df_shops.head()


# In[21]:


df_shops.drop(columns=['shop_name', 'city_name'], inplace=True)
df_shops.head()


# In[22]:


df_shops.info()


# In[23]:


df_shops.describe()


# ## Items and Categories

# In[24]:


df_items = pd.read_csv(
    input_prefix + '../items_full' + '.csv' + bzfile,
    dtype={'item_category_id': 'category', 'subject_code': 'category',
           'group_code': 'category', 'subgroup_code': 'category'},
    index_col='item_id')
print('shape:', df_items.shape)
df_items.head()


# In[25]:


df_items.drop(columns=['item_name', 'subject_name', 'item_category_name', 'group_name', 'subgroup_name'], inplace=True)
df_items.head()


# In[26]:


df_items.info()


# In[27]:


df_items.describe()


# ## Sales (full)

# In[28]:


df_train2.head()


# In[29]:


df_train2.reset_index().head(10)


# In[30]:


#shop_id	item_id	date_block_num


# In[ ]:





# In[ ]:





# In[31]:


df_train3 = df_train2.join(df_shops).join(df_items)
df_train3.head()


# In[32]:


df_train3.head(10)


# In[33]:


df_train3.to_csv('train_full.csv')


# In[34]:


get_ipython().system('head train_full.csv')


# In[35]:


get_ipython().system('rm -f train_full.csv.bz2 && bzip2 -9 train_full.csv')


# ## Sales (monthly full)

# In[36]:


df_test.head()


# In[37]:


df_monthly.head()


# In[38]:


df_monthly2 = df_monthly.join(df_shops).join(df_items)
df_monthly2.head()


# In[39]:


df_monthly2.to_csv('sales_monthly_full.csv')


# In[40]:


get_ipython().system('rm -f sales_monthly_full.csv.bz2 && bzip2 -9 sales_monthly_full.csv')

