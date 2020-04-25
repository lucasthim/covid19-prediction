import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,recall_score, classification_report, auc, roc_curve,roc_auc_score

plt.style.use('seaborn')

def predict(model,X,y):
    df_result = pd.DataFrame(columns = ['TrueClass','Predicted'])
    df_result.Predicted = model.predict(X.values)
    df_result.TrueClass = y.values.ravel()
    return df_result

def find_best_classification_model_with_cross_validation(model,parameters,X_train,y_train,k_folds = 10,metric = 'f1',verbose = 0):
    start = time.time()
    grid_search = GridSearchCV(
        estimator = model,
        param_grid = parameters,
        cv = k_folds,
        scoring = metric, 
        verbose = verbose,
        n_jobs = -1)
    grid_search.fit(X_train,y_train)

    if verbose > 0:
        print("--- Ellapsed time: %s seconds ---" % (time.time() - start))
        print('Best params: ',grid_search.best_params_)
        print('Best score (%s)' % metric,grid_search.best_score_)
    return grid_search.best_estimator_,grid_search.best_params_, grid_search.best_score_

def evalute_model_performance(model,model_name,X,y,df_result):
    plot_confusion_matrix(df_result,model_name)
    acc_report = classification_report(df_result.TrueClass, df_result.Predicted,target_names =['Negative', 'Positive'])
    print(acc_report)
    try:
        plot_ROC(model, model_name, X, y)
    except:
        print('Could not print ROC AUC curve.')


def plot_confusion_matrix(df,title,labels = ['Negative', 'Positive'],dataset_type = 'Validation'):
    conf_matrix = confusion_matrix(df.TrueClass, df.Predicted)
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d");
    plt.title('{0} - Confusion matrix - {1} set'.format(title,dataset_type), fontsize = 20)
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.show()
    return conf_matrix.ravel()
    

def plot_ROC(model,model_name,X_test,y_test):
    
    naive_probs = [0 for _ in range(len(y_test))]
    
    probs = model.predict_proba(X_test)
    probs = probs[:, 1]

    naive_auc = roc_auc_score(y_test, naive_probs)
    model_auc = roc_auc_score(y_test, probs)

    print('No Skill: ROC AUC=%.3f' % (naive_auc))
    print(model_name,': ROC AUC=%.3f' % (model_auc))

    naive_fpr, naive_tpr, _ = roc_curve(y_test, naive_probs)
    model_fpr, model_tpr, _ = roc_curve(y_test, probs)
    
    plt.plot(naive_fpr, naive_tpr, linestyle='--', label='Naive')
    plt.plot(model_fpr, model_tpr, marker='.', label=model_name)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    