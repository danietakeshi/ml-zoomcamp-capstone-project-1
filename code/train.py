import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
import xgboost as xgb

eta = 0.1
max_depth = 6
alpha = 10
lambda_ = 10
gamma = 10

print('Loading the data...')

df = pd.read_csv('../data/BankChurners.csv')

print('Formatting the data...')

df.columns = df.columns.str.lower()

df = df.drop(columns=['clientnum'])

column_filter = df.dtypes.index[df.dtypes.values != 'object'].values
for column in column_filter:
    if '_ct' in column or 'avg' in column or 'naive_bayes_' in column:
        df = df.drop(columns=[column])

df.attrition_flag = (df.attrition_flag == 'Attrited Customer').astype(int)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

columns = ['height', 'weight', 'density', 'bmi', 'diameter', 'length', 'sex']

y_full_train = df_full_train.attrition_flag.values
y_test = df_test.attrition_flag.values

del df_full_train['attrition_flag']
del df_test['attrition_flag']

columns = [
        'customer_age',
        'gender',
        'dependent_count',
        'education_level',
        'marital_status',
        'income_category',
        'card_category',
        'months_on_book',
        'total_relationship_count',
        'months_inactive_12_mon',
        'contacts_count_12_mon',
        'credit_limit',
        'total_trans_amt',
    ]

dv = DictVectorizer(sparse = False)

X_full_train = dv.fit_transform(df_full_train[columns].to_dict(orient='records'))
X_test = dv.transform(df_test[columns].to_dict(orient='records'))

features = dv.get_feature_names_out().tolist()

dtrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)
dval = xgb.DMatrix(X_test, label=y_test, feature_names=features)

xgb_params = {
    'eta': eta, 
    'max_depth': max_depth,
    'min_child_weight': 1,
    'alpha': alpha,
    'lambda': lambda_,
    'gamma': gamma,
    
    'objective': 'binary:logistic',
    'nthread': 3,
    
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=100, verbose_eval=5)

y_pred = model.predict(dval)


model_validation = xgb.cv(xgb_params, 
               dtrain,
               num_boost_round=100,
               verbose_eval=10,
               nfold=100,
               metrics="auc",)

output_file = f'model_eta={eta}_max_depth={max_depth}_v{model_validation["test-auc-mean"].mean():.3f}.bin'

print(f'Saving the model on {output_file}')

with open(output_file, 'wb') as f_out: 
    pickle.dump((dv, model, features), f_out)

