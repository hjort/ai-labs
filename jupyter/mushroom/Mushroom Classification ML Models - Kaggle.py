
# coding: utf-8

# ## Mushroom Classification - Kaggle
# 
# - https://www.kaggle.com/nirajvermafcb/comparing-various-ml-models-roc-curve-comparison

# In[1]:


# Importing all the libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("mushrooms/mushrooms.csv")
data.head(6)


# In[3]:


# Let us check if there is any null values
data.isnull().sum()


# In[4]:


data['class'].unique()

# Thus we have two claasification. Either the mushroom is poisonous or edible


# In[5]:


data.shape

# Thus we have 22 features (1st one is label) and 8124 instances.


# In[6]:


# We can see that the dataset has values in strings.
# We need to convert all the unique values to integers.
# Thus we perform label encoding on the data.

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])
 
data.head()


# In[7]:


# Checking the encoded values

data['stalk-color-above-ring'].unique()


# In[8]:


print(data.groupby('class').size())


# In[9]:


# Plotting boxplot to see the distribution of the data

'''
# Create a figure instance
fig, axes = plt.subplots(nrows=2 ,ncols=2 ,figsize=(9, 9))

# Create an axes instance and the boxplot
bp1 = axes[0,0].boxplot(data['stalk-color-above-ring'],patch_artist=True)

bp2 = axes[0,1].boxplot(data['stalk-color-below-ring'],patch_artist=True)

bp3 = axes[1,0].boxplot(data['stalk-surface-below-ring'],patch_artist=True)

bp4 = axes[1,1].boxplot(data['stalk-surface-above-ring'],patch_artist=True)
'''

ax = sns.boxplot(x='class', y='stalk-color-above-ring', data=data)
ax = sns.stripplot(x="class", y='stalk-color-above-ring', data=data, jitter=True, edgecolor="gray")

#sns.plt.title("Class w.r.t stalkcolor above ring", fontsize=12)


# In[10]:


# Separating features and label

X = data.iloc[:,1:23]  # all rows, all the features and no labels
y = data.iloc[:, 0]  # all rows, label only

print(X.shape)
print(y.shape)


# In[11]:


X.head()


# In[12]:


y.head()


# In[13]:


X.describe()


# In[14]:


data.corr()


# In[15]:


# Standardising the data

# Scale the data to be between -1 and 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X = scaler.fit_transform(X)

print(X)


# In[16]:


# Principal Component Analysis (PCA)

from sklearn.decomposition import PCA
pca = PCA()

pca.fit_transform(X)

# Note: We can avoid PCA here since the dataset is very small.


# In[17]:


covariance = pca.get_covariance()
covariance


# In[18]:


explained_variance = pca.explained_variance_
explained_variance


# In[19]:


with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))
    plt.bar(range(22), explained_variance, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    
# We can see that the last 4 components has less amount of variance of the data.
# The 1st 17 components retains more than 90% of the data.


# In[20]:


# Let us take only first two principal components and visualise it using K-means clustering

N = data.values
pca = PCA(n_components=2)
x = pca.fit_transform(N)
plt.figure(figsize = (5,5))
plt.scatter(x[:,0],x[:,1])
plt.show()


# In[21]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=5)
X_clustered = kmeans.fit_predict(N)

LABEL_COLOR_MAP = {0 : 'g',
                   1 : 'y'
                  }

label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]
plt.figure(figsize = (5,5))
plt.scatter(x[:,0],x[:,1], c= label_color)
plt.show()

# Thus using K-means we are able segregate 2 classes well using the first two components with maximum variance.


# In[22]:


# Performing PCA by taking 17 components with maximum Variance

pca_modified = PCA(n_components=17)
pca_modified.fit_transform(X)


# In[23]:


# Splitting the data into training and testing dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[24]:


# 1) Default Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics

model_LR = LogisticRegression()
model_LR.fit(X_train, y_train)


# In[25]:


# This will give you positive class prediction probabilities
y_prob = model_LR.predict_proba(X_test)[:,1]

# This will threshold the probabilities to give class predictions
y_pred = np.where(y_prob > 0.5, 1, 0)

model_LR.score(X_test, y_pred)


# In[26]:


confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
confusion_matrix


# In[27]:


auc_roc = metrics.roc_auc_score(y_test, y_pred)
auc_roc


# In[28]:


from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[29]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# In[30]:


# 2) Logistic Regression (Tuned model)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics

LR_model = LogisticRegression()

tuned_parameters = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l1','l2']
}


# - L1 and L2 are regularization parameters. They're used to avoid overfiting. Both L1 and L2 regularization prevents overfitting by shrinking (imposing a penalty) on the coefficients.
# - L1 is the first moment norm |x1-x2| (|w| for regularization case) that is simply the absolute dıstance between two points where L2 is second moment norm corresponding to Euclidean Distance that is |x1-x2|^2 (|w|^2 for regularization case).
# - In simple words, L2 (Ridge) shrinks all the coefficient by the same proportions but eliminates none, while L1 (Lasso) can shrink some coefficients to zero, performing variable selection.
# - If all the features are correlated with the label, ridge outperforms lasso, as the coefficients are never zero in ridge.
# - If only a subset of features are correlated with the label, lasso outperforms ridge as in lasso model some coefficient can be shrunken to zero.

# In[31]:


# Taking a look at the correlation
data.corr()


# In[32]:


# The grid search provided by GridSearchCV exhaustively generates candidates from a grid of
# parameter values specified with the tuned_parameter. The GridSearchCV instance implements
# the usual estimator API: when “fitting” it on a dataset all the possible combinations of
# parameter values are evaluated and the best combination is retained.

from sklearn.model_selection import GridSearchCV

LR = GridSearchCV(LR_model, tuned_parameters, cv=10)

LR.fit(X_train, y_train)


# In[33]:


print(LR.best_params_)


# In[34]:


# This will give you positive class prediction probabilities
y_prob = LR.predict_proba(X_test)[:,1]

# This will threshold the probabilities to give class predictions
y_pred = np.where(y_prob > 0.5, 1, 0)

LR.score(X_test, y_pred)


# In[35]:


confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
confusion_matrix


# In[37]:


auc_roc = metrics.classification_report(y_test, y_pred)
print(auc_roc)


# In[38]:


auc_roc = metrics.roc_auc_score(y_test, y_pred)
auc_roc


# In[39]:


from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[40]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# In[41]:


LR_ridge = LogisticRegression(penalty='l2')

LR_ridge.fit(X_train, y_train)


# In[42]:


y_prob = LR_ridge.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
LR_ridge.score(X_test, y_pred)


# In[43]:


confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[44]:


auc_roc=metrics.classification_report(y_test,y_pred)
print(auc_roc)


# In[45]:


auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[46]:


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[47]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# In[48]:


# 3) Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB
model_naive = GaussianNB()
model_naive.fit(X_train, y_train)


# In[49]:


y_prob = model_naive.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
model_naive.score(X_test, y_pred)


# In[50]:


print("Number of mislabeled points from %d points : %d"
      % (X_test.shape[0],(y_test!= y_pred).sum()))


# In[51]:


scores = cross_val_score(model_naive, X, y, cv=10, scoring='accuracy')
print(scores)


# In[52]:


scores.mean()


# In[53]:


confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[54]:


auc_roc=metrics.classification_report(y_test,y_pred)
print(auc_roc)


# In[55]:


auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[56]:


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[57]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# In[59]:


# 4) Support Vector Machine

from sklearn.svm import SVC
svm_model = SVC()


# In[60]:


# 4.1) Support Vector Machine without polynomial kernel

tuned_parameters = {
 'C': [1, 10, 100,500, 1000], 'kernel': ['linear','rbf'],
 'C': [1, 10, 100,500, 1000], 'gamma': [1,0.1,0.01,0.001, 0.0001], 'kernel': ['rbf'],
 #'degree': [2,3,4,5,6] , 'C':[1,10,100,500,1000] , 'kernel':['poly']
}

from sklearn.grid_search import RandomizedSearchCV

model_svm = RandomizedSearchCV(svm_model, tuned_parameters, cv=10, scoring='accuracy', n_iter=20)


# In[61]:


model_svm.fit(X_train, y_train)
print(model_svm.best_score_)


# In[62]:


print(model_svm.grid_scores_)


# In[63]:


print(model_svm.best_params_)


# In[64]:


y_pred= model_svm.predict(X_test)
print(metrics.accuracy_score(y_pred,y_test))


# In[65]:


confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[66]:


auc_roc=metrics.classification_report(y_test,y_pred)
print(auc_roc)


# In[67]:


auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[68]:


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[69]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# In[70]:


# 4.2) Support Vector machine with polynomial Kernel

tuned_parameters = {
 'C': [1, 10, 100,500, 1000], 'kernel': ['linear','rbf'],
 'C': [1, 10, 100,500, 1000], 'gamma': [1,0.1,0.01,0.001, 0.0001], 'kernel': ['rbf'],
 'degree': [2,3,4,5,6] , 'C':[1,10,100,500,1000] , 'kernel':['poly']
}

# ...


# In[71]:


# 5) Trying default model

from sklearn.ensemble import RandomForestClassifier

model_RR = RandomForestClassifier()

#tuned_parameters = {'min_samples_leaf': range(5,10,5), 'n_estimators' : range(50,200,50),
                    #'max_depth': range(5,15,5), 'max_features':range(5,20,5)
                    #}


# In[72]:


model_RR.fit(X_train, y_train)


# In[73]:


# ...


# In[74]:


# 6) Default Decision Tree model

from sklearn.tree import DecisionTreeClassifier

model_tree = DecisionTreeClassifier()


# In[75]:


model_tree.fit(X_train, y_train)


# In[76]:


# ...


# In[77]:


# 7) Neural Network

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier()

mlp.fit(X_train, y_train)

