#!/usr/bin/env python
# coding: utf-8

# # Predict Future Sales in Russia
# 
# - https://www.kaggle.com/c/competitive-data-science-predict-future-sales/

# ### Data files
# - item_categories.csv
# - items.csv
# - sales_train.csv
# - sample_submission.csv
# - shops.csv
# - test.csv

# In[2]:


# importar pacotes necessários
import numpy as np
import pandas as pd


# In[3]:


# definir parâmetros extras
pd.set_option('precision', 4)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


# In[3]:


input_prefix = 'data/'
#input_prefix = 'https://github.com/hjort/ai-labs/raw/master/jupyter/future-sales/data/'

translate = False # original em russo
#translate = True # traduzir para inglês

trfile = '-translated' if translate else ''
trcol = '_translated' if translate else ''

bzfile = '.bz2' # ''


# ## Shops

# In[4]:


df_shops = pd.read_csv(
    input_prefix + 'shops' + trfile + '.csv' + bzfile,
    index_col='shop_id')
print('shape:', df_shops.shape)
df_shops.head()


# In[5]:


df_shops.dtypes


# In[6]:


df_shops.describe()


# In[7]:


df_shops.info()


# ### Extract city name from the shop name
# 
# - 'Москва ТРК "Атриум"' => 'Москва'
# - 'Н.Новгород ТРЦ "Фантастика" => 'Н.Новгород'

# In[8]:


df_shops['city_name'] = df_shops['shop_name' + trcol].apply(lambda s: s.split()[0])

df_shops.head()


# In[9]:


df_shops.groupby('city_name').count().    sort_values(by='shop_name', ascending=False).head()


# In[10]:


df_shops['city_code'] = (
    df_shops['city_name'].astype('category').cat.codes + 1
).astype('category')

df_shops.head()


# In[11]:


df_shops.info()


# In[12]:


df_shops.describe()


# In[13]:


df_shops.to_csv('shops_full.csv')


# In[14]:


get_ipython().system('head shops_full.csv')


# In[15]:


get_ipython().system('rm -f shops_full.csv.bz2 && bzip2 -9 shops_full.csv')


# ## Item categories

# In[16]:


df_categories = pd.read_csv(
    input_prefix + 'item_categories' + trfile + '.csv' + bzfile,
    index_col='item_category_id')
print('shape:', df_categories.shape)
df_categories.head()


# In[17]:


df_categories.describe()


# ### Extract group and subgroup names from item category name
# 
# - 'Игровые консоли - PS4' => 'Игровые консоли'
# - 'Карты оплаты - Windows (Цифра)' => 'Карты оплаты'
# - 'Книги - Комиксы, манга' => 'Книги'

# In[18]:


df_categories['group_name'] =     df_categories['item_category_name' + trcol].apply(
        lambda s: s.split(' - ')[0].split(' (')[0].upper())

df_categories.head()


# In[19]:


df_categories.groupby('group_name').count().    sort_values(by='item_category_name', ascending=False).head()


# In[20]:


def extract_subgroup(s):
    gs = s.split(' - ')
    if len(gs) > 1:
        gs2 = gs[1].split(' (')
        return gs2[0].upper()
    else:
        return ''
    #return gs[1] if len(gs) > 1 else ''
    
df_categories['subgroup_name'] =     df_categories['item_category_name' + trcol].apply(
        lambda s: extract_subgroup(s))

df_categories.head()


# In[21]:


df_categories.groupby('subgroup_name').count().head(10)


# In[22]:


df_categories['group_code'] = (
    df_categories['group_name'].astype('category').cat.codes + 1).astype('category')

df_categories['subgroup_code'] = (
    df_categories['subgroup_name'].astype('category').cat.codes + 1).astype('category')

df_categories.head()


# In[23]:


df_categories.info()


# In[24]:


df_categories.describe()


# ## Items

# In[25]:


df_items = pd.read_csv(
    input_prefix + 'items' + trfile + '.csv' + bzfile,
    index_col='item_id', dtype={'item_category_id': np.int8})
print('shape:', df_items.shape)
df_items.head()


# In[26]:


#if not translate:
#    df_items['item_category_id'] = df_items['item_category_id'].astype('category')


# In[27]:


df_items.describe()


# In[28]:


import re

def extract_main_subject(str):
    s = str.upper()
    # remover caracteres do começo => !"*/
    s = re.sub("^[!*/\"]+ ?", "", s)
    # remover termo "1C:" do começo do nome
    s = re.sub("^1C.", "", s)
    # remover termo "THE" do começo do nome
    s = re.sub("^THE ", "", s)
    # obter primeira palavra em maiúsculo
    s = s.split()[0]
    # substituir caracteres => '`’
    s = re.sub("['`’]", "_", s)
    # remover caracteres do fim da palavra => :.®,!
    s = re.sub("[:.,!®]$", "", s)
    return s
    
df_items['subject_name'] =     df_items['item_name' + trcol].apply(
        lambda s: extract_main_subject(s))

df_items.head()


# In[29]:


df_items.tail()


# In[30]:


df_items.groupby('subject_name').count().head(10)


# In[31]:


df_items['subject_code'] = (
    df_items['subject_name'].astype('category').cat.codes + 1).astype('category')

df_items.head()


# In[32]:


df_items.tail()


# In[33]:


df_items.info()


# In[34]:


df_items.describe()


# In[35]:


# join items + categories:item_category_id => group_code, subgroup_code
df_items2 = pd.merge(df_items, #.reset_index(),
                     df_categories, how='left', on='item_category_id')
df_items2.index.names = ['item_id']
df_items2.head()


# In[36]:


del(df_items)
del(df_categories)


# In[37]:


#df_items2.set_index(['item_id'], inplace=True)
df_items2['item_category_id'] = df_items2['item_category_id'].astype('category')


# In[38]:


df_items2.info()


# In[39]:


df_items2.describe()


# In[40]:


df_items2.to_csv('items_full.csv')


# In[41]:


get_ipython().system('head items_full.csv')


# In[42]:


get_ipython().system('rm -f items_full.csv.bz2 && bzip2 -9 items_full.csv')


# ## Date Blocks

# In[34]:


# criar dataframe com possíveis meses
df_dates = pd.DataFrame({'last_date': pd.date_range(start='2013-01-01', periods=35, freq='M')})
df_dates.index.names = ['date_block_num']
df_dates['first_date'] = df_dates['last_date'].values.astype('datetime64[M]')
df_dates['days_count'] = ((df_dates['last_date'] - df_dates['first_date']).dt.days + 1).astype(np.int8)
df_dates['year'] = df_dates['last_date'].dt.year
df_dates['month'] = df_dates['last_date'].dt.month
df_dates.head()


# In[35]:


df_dates.tail()


# In[36]:


df_dates.info()

# criar colunas ano e mês
df_dates['year'] = df_dates['date'].dt.year
df_dates['month'] = df_dates['date'].dt.month
df_dates['last_day'] = df_dates['date'].dt.day
df_dates.head()
# In[37]:


#TODO: criar colunas sun_cnt, mon_cnt, ..., sat_cnt

from datetime import date
dt = date(2013, 5, 1)
dtdef count_weekdays(year, month, last_day):
    weekdays = np.zeros((7,), dtype=np.int8)
    for day in range(1, last_day + 1):
        dt = date(year, month, day)
        wd = dt.weekday()
        weekdays[wd] += 1
        #print(dt, wd)
    return weekdays

#wd = count_weekdays(2013, 1, 31)
#print(wd)#df_dates['sun_cnt'] = df_dates.apply()
df_dates.apply(lambda x: count_weekdays(x.year, x.month, x.last_day))
# In[38]:


df_dates.to_csv('date_blocks.csv')


# In[39]:


get_ipython().system('head date_blocks.csv')


# In[ ]:




