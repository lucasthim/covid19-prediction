# Interpretable Machine Learning for COVID-19 Diagnosis Through Clinical Variables

Predict if a patient has the SarsCov-2 virus based on some clinical variables.

Our goal is to propose an efficient and also transparent/interpretable ML solution to the correct predicition of suspicious covid-19 cases.

Secondary goal will be to eliminate features while keeping the preditive power of the solution.

Classification algorithms utilized in the solution:
- Random Forest
- Logistic Regression
- Support Vector Machine

Interpretation algorithms:
- SHAP
- Regression weights

Original dataset comes from a Kaggle Competition held by Einstein Data4u. It can be found at: 
https://www.kaggle.com/dataset/e626783d4672f182e7870b1bbe75fae66bdfb232289da0a61f08c2ceb01cab01/tasks?taskId=645

Specific steps adopted on this novel:
- Remove the least amount of rows and columns in order to eliminate "holes" of missing values in the dataset.
- Execute a Grid Search with Cross Validation to tune models hyperparameters and obtain preliminare results.
- Execute a Recursive Feature Elimination (RFE) with Cross Validation. Test models after RFE-CV without tuning hyper parameters.
- Choose variables to eliminate and execute another Grid Search with remaining variables.
- Visualize feature weights with feature_importance of linear models and SHAP values of non-linear models.

# About the original paper

Link: https://doi.org/10.48011/asba.v2i1.1590

Abstract:

This work proposes an interpretable machine learning approach to diagnose suspected COVID-19 cases based on clinical variables. Results obtained for the proposed models have F-2 measure superior to 0.80 and accuracy superior to 0.85. Interpretation of the linear model feature importance brought insights about the most relevant features. Shapley Additive Explanations were used in the non-linear models. They were able to show the difference between positive and negative patients as well as offer a global interpretability sense of the models.


If you enjoy this work, please cite as :

@article{thimoteo_vellasco_amaral_figueiredo_yokoyama_marques_2020, title={Interpretable Machine Learning for COVID-19 Diagnosis Through Clinical Variables}, DOI={10.48011/asba.v2i1.1590}, journal={Anais do Congresso Brasileiro de Automática 2020}, author={Thimoteo, Lucas M. and Vellasco, Marley M. and Amaral, Jorge M. Do and Figueiredo, Karla and Yokoyama, Cátia Lie and Marques, Erito}, year={2020}, month={Dec}}
