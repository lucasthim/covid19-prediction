# COVID-19 Diagnosis Through Clinical Variables
Predict if a patient has the SarsCov-2 virus based on some clinical variables.

Original dataset comes from a Kaggle Competition held by Einstein Data4u. It can be found at: 
https://www.kaggle.com/dataset/e626783d4672f182e7870b1bbe75fae66bdfb232289da0a61f08c2ceb01cab01/tasks?taskId=645


My goal is to propose an efficient and also transparent/interpretable ML solution to the correct predicition of suspicious covid-19 cases.

Secondary goal will be to eliminate features while keeping the preditive power of the solution.

Classification algorithms utilized in the solution:
- Random Forest
- Logistic Regression
- Support Vector Machine

This novel consists on the following approach:
- Remove the least amount of rows and columns in order to eliminate "holes" of missing values in the dataset.
- Execute a Grid Search with Cross Validation to tune models hyperparameters and obtain preliminare results.
- Visualize feature weights with feature_importance of linear models and SHAP values of non-linear models.
- Execute a Recursive Feature Elimination (RFE) with Cross Validation. Test models after RFE-CV without tuning hyper parameters.
- Visualize feature weights with feature_importance of linear models and SHAP values of non-linear models.
- Choose variables to eliminate and execute another Grid Search with remaining variables
- Visualize feature weights with feature_importance of linear models and SHAP values of non-linear models.


Next steps:
- Impute missing values with the Miss Forest algorithm.
- Optimization with some Evolutionary Algorithm (PSO, GA or DE)
