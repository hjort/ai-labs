
# coding: utf-8

# ## Linear Regression on Boston Housing Dataset - TDS
# 
# Extraído de https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155

# In[1]:


# import the required libraries
import numpy as np
import matplotlib.pyplot as plt 

import pandas as pd  
import seaborn as sns 

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# load the housing data from the scikit-learn library
from sklearn.datasets import load_boston
boston_dataset = load_boston()


# In[8]:


# understand what it contains
print(boston_dataset.keys())


# In[4]:


print(boston_dataset.DESCR)


# In[5]:


# load the data into a pandas dataframe
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()


# In[9]:


# The prices of the house indicated by the variable MEDV is our target variable and the
# remaining are the feature variables based on which we will predict the value of a house.

# create a new column of target values and add it to the dataframe
boston['MEDV'] = boston_dataset.target


# In[10]:


# see if there are any missing values in the data

# count the number of missing values for each feature
boston.isnull().sum()

# (there are no missing values in this dataset)


# In[11]:


# Exploratory Data Analysis is a very important step before training the model

# use some visualizations to understand the relationship of the target variable with other features

# plot the distribution of the target variable MEDV
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(boston['MEDV'], bins=30)
plt.show()

# (the values of MEDV are distributed normally with few outliers)


# In[12]:


# create a correlation matrix that measures the linear relationships between the variables

correlation_matrix = boston.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)

# The correlation coefficient ranges from -1 to 1. If the value is close to 1, it means that
# there is a strong positive correlation between the two variables. When it is close to -1,
# the variables have a strong negative correlation.


# ### Observations:
# - To fit a linear regression model, we select those features which have a high correlation with our target variable MEDV. By looking at the correlation matrix we can see that RM has a strong positive correlation with MEDV (0.7) where as LSTAT has a high negative correlation with MEDV(-0.74).
# - An important point in selecting features for a linear regression model is to check for multi-co-linearity. The features RAD, TAX have a correlation of 0.91. These feature pairs are strongly correlated to each other. We should not select both these features together for training the model. Same goes for the features DIS and AGE which have a correlation of -0.75.

# In[13]:


# Based on the above observations we will RM and LSTAT as our features.

# Using a scatter plot let’s see how these features vary with MEDV.

plt.figure(figsize=(20, 5))

features = ['LSTAT', 'RM']
target = boston['MEDV']

for i, col in enumerate(features):
    plt.subplot(1, len(features), i + 1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')


# ### Observations:
# - The prices increase as the value of RM increases linearly. There are few outliers and the data seems to be capped at 50.
# - The prices tend to decrease with an increase in LSTAT. Though it doesn’t look to be following exactly a linear line.

# In[14]:


# concatenate the LSTAT and RM columns using np.c_ provided by the numpy library
X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT', 'RM'])
Y = boston['MEDV']


# In[15]:


X.head()


# In[16]:


Y.head()


# In[17]:


# split the data into training and testing sets
from sklearn.model_selection import train_test_split

# We train the model with 80% of the samples and test with the remaining 20%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[19]:


X_train.describe()


# In[20]:


Y_train.describe()


# In[28]:


# use scikit-learn’s LinearRegression to train our model on both the training and test sets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)


# In[29]:


lin_model.coef_


# In[30]:


lin_model.intercept_


# In[31]:


# evaluate our model using RMSE and R2-score

# model evaluation for training set
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# In[32]:


# plotting the y_test vs y_pred
# ideally should have been a straight line
plt.scatter(Y_test, y_test_predict)
plt.show()

