import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pandas import set_option
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import plot_confusion_matrix
from sklearn.externals import joblib
from pickle import dump
from pickle import load



# cor_grid_ml: Use GridSearchCV to find the best parameters for the ml models
# after you locate the best parameter region with the randomized search, you can perform a local search
# ver 0.1 : Jorge Amaral

program_name = "cor_grid_ml.py"

model_number = 4 # 0:LR 1:RF 2:ADAB 3:XGB 4:LGB

# load dataset
train_csv = "train_class_data.csv"
data_train = pd.read_csv(train_csv, encoding='utf-8', sep=',')
print((data_train.dtypes))
ncols = data_train.shape[1]  # get the number of columns
array = data_train.values


X_train = array[:, 3: -1]
Y_train = array[:, -1]
number_of_features = [ncols - 2]

test_csv = "test_class_data.csv "
data_test = pd.read_csv(test_csv, encoding='utf-8', sep=',')

array = data_test.values
X_test = array[:, 3: -1]
Y_test = array[:, -1]


#  define model: choose the model number
#  you can also put the parameters values in the if for each model


# Create Model
if model_number == 0:
    model_name = "LR"
    param_search = {
        'LR__C': [44, 45, 46, 47, 48, 49],
        'LR__penalty': ['l2']
    }

    LR = LogisticRegression(random_state=101)

    # create pipeline
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('LR', LogisticRegression(random_state=101)))
    model = Pipeline(estimators)

elif model_number == 1:
    model_name = "RF"
    param_search = {
        'RF__n_estimators': [360, 370, 380],
        'RF__max_depth': [3, 4, 5]
    }
    # create pipeline
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('RF', RandomForestClassifier(random_state=101)))
    model = Pipeline(estimators)

elif model_number == 2:
    model_name = "ADAB"
    param_search = {"ADAB__base_estimator__max_depth": [7, 8, 9],
                    "ADAB__n_estimators": [710, 720, 730]
                    }

    DTC = DecisionTreeClassifier(max_features="auto", class_weight="balanced", max_depth=None, random_state=101)

    # create pipeline
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('ADAB', AdaBoostClassifier(base_estimator=DTC, random_state=101)))
    model = Pipeline(estimators)
elif model_number == 3:
    model_name = "XGB"
    param_search = {'XGB__max_depth': [5, 6, 7],
                    'XGB__n_estimators': [280, 300, 320],
                    'XGB__learning_rate': [0.1],
                    'XGB__gamma': [0]
                    }

    # create pipeline
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('XGB', XGBClassifier(booster='gbtree', random_state=101)))
    model = Pipeline(estimators)
elif model_number == 4:
    # Lightgbm
    model_name = "LGBM"
    param_search = {'LGBM__num_leaves': [4, 5, 6],
                    'LGBM__n_estimators': [90, 100, 110],
                    'LGBM__boosting_type': ['dart'],
                    'LGBM__class_weight': [None]
                    }

    # create pipeline
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('LGBM', LGBMClassifier(random_state=101, max_depth=0)))
    model = Pipeline(estimators)

# elif model_number ==5:
#
# else:


# define n_iter ( number of models) and cv (number of crossvalidation) and scoring

n_iter = 10
cv = 3
# scoring = make_scorer(fbeta_score, beta=0.5)
# scoring = 'roc_auc'
# scoring = 'f1'
scoring = 'accuracy'

# search object

search = GridSearchCV(estimator=model, param_grid=param_search, scoring=scoring, cv=cv, verbose=1)

start_time = time.time()
search_result = search.fit(X_train, Y_train)
print("--- %s seconds ---" % (time.time() - start_time))

# Write the title for the confusion matrix
tag = "Model: %s / Best: %f (%s) using %s" % (model_name, search_result.best_score_, scoring, search_result.best_params_)
print(tag)
y_pred = search_result.predict(X_test)

classes = data_train.Event.unique().tolist()
cm = confusion_matrix(Y_test, y_pred, labels = classes)
plot_confusion_matrix(cm, classes,title  = tag)

# save file with pickle
dump(search_result, open('BestClassifierModel.pkl', 'wb'))

print(search_result.classes_)

loaded_model = load(open('BestClassifierModel.pkl', 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)

plt.savefig(model_name + '_confusion_matrix.png')
plt.show()


# Add the prediction to the test dataframe
data_test['predict'] = search_result.predict(X_test)


# Calculate the error and save to csv and excel files
error = data_test[data_test['predict'] != data_test['Event']]
error.to_csv(model_name + '_test_data_errors.csv')
error.to_excel(model_name + '_test_data_errors.xls')

#Save to GEOJSON
to_geojson(error,model_name)

#
print("End of %s" % (program_name))
