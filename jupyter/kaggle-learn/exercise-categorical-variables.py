####################################################################################################

### Load Train and Test Data

X = pd.read_csv('../input/train.csv', index_col='Id') 
X_test = pd.read_csv('../input/test.csv', index_col='Id')

X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice

####################################################################################################

### Handle Missing Values

cols_with_missing = list(
  set([col for col in X.columns if X[col].isnull().any()]) |
  set([col for col in X_test.columns if X_test[col].isnull().any()]))
print('Columns with missing values:\n', cols_with_missing)

missing_val_count_by_column = (X.isnull().sum()).to_frame()
print('\nMissing values per column:\n', missing_val_count_by_column[missing_val_count_by_column > 0])

#TODO: calcular percentual de valores faltantes por coluna
total_rows_count = X.shape[0] + X_test.shape[0]

#TODO: remover colunas com valores faltantes em < 80% dos casos
#TODO: imputar valores faltantes nas colunas com >= 80% dos casos

threshold = 0.8
cols_missing_drop = ...
cols_missing_impute = set(cols_with_missing) - set(cols_missing_drop)

####################################################################################################

X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)

####################################################################################################

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer(strategy='most_frequent')
imputed_X = pd.DataFrame(my_imputer.fit_transform(X))
imputed_X_test = pd.DataFrame(my_imputer.transform(X_test))

imputed_X.columns = X.columns
imputed_X_test.columns = X_test.columns

####################################################################################################

missing_val_count_by_column = (imputed_X_test.isnull().sum())
print('\nNumber of missing values per column:\n', missing_val_count_by_column[missing_val_count_by_column > 0])

####################################################################################################

X = imputed_X
X_test = imputed_X_test

####################################################################################################

### Handle Categorical Variables

object_cols = [col for col in X.columns if X[col].dtype == "object"]
good_label_cols = [col for col in object_cols if set(X[col]) == set(X_test[col])]
bad_label_cols = list(set(object_cols) - set(good_label_cols))

print('All categorical columns in the dataset:', object_cols)
print('\nCategorical columns that will be label encoded:', good_label_cols)
print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)

####################################################################################################

from sklearn.preprocessing import LabelEncoder

label_X = X.drop(bad_label_cols, axis=1)
label_X_test = X_test.drop(bad_label_cols, axis=1)

label_encoder = LabelEncoder()
for col in good_label_cols:
    label_X[col] = label_encoder.fit_transform(X[col])
    label_X_test[col] = label_encoder.transform(X_test[col])

####################################################################################################

object_nunique = list(map(lambda col: X[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

nunique = sorted(d.items(), key=lambda x: x[1])
print('Number of unique entries in each column with categorical data:', nunique)

####################################################################################################

low_cardinality_cols = [col for col in object_cols if X[col].nunique() < 10]
high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))
#high_cardinality_cols = [k for k, v in d.items() if v > 10]

print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)

####################################################################################################

from sklearn.preprocessing import OneHotEncoder

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(X[low_cardinality_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[low_cardinality_cols]))

OH_cols.index = X.index
OH_cols_test.index = X_test.index

num_X = X.drop(object_cols, axis=1)
num_X_test = X_test.drop(object_cols, axis=1)

OH_X = pd.concat([num_X, OH_cols], axis=1)
OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)

####################################################################################################

final_X = OH_X.drop(['SalePrice'], axis=1)
final_y = OH_X['SalePrice']
final_X_test = OH_X_test

####################################################################################################

#model = RandomForestRegressor(n_estimators=200, random_state=0)
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=4)

model.fit(final_X, final_y)
preds_test = model.predict(final_X_test)

####################################################################################################

output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)

####################################################################################################

