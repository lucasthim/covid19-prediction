import shap
import pandas as pd
import numpy as np

class SHAPEstimatorWrapper():

        '''
        SHAP Wrapper on estimator. 
        
        This class aims to offer the coefficients (coef_) of any estimator as shapley values.

        The coefficients can be used as feature weights and help on feature selection or feature extraction.

        Tip: passing on the SHAP object prior fitting the estimator might lead to faster calculation of coefficients.

        '''

        def __init__(self, estimator, estimator_type = None, use_shap = True):
            self.estimator = estimator
            self.estimator_type = estimator_type
            self.shap_coef_ = None
            self.shap_values = None
            self.explainer = None
            self.use_shap = use_shap
            self.X = None

        def fit(self, *args,**kwargs):
            self.estimator.fit(*args,**kwargs)
            self.X = args[0]

        def predict(self, *args, **kwargs):
            return self.estimator.predict(*args,**kwargs)
        
        def predict_proba(self, *args, **kwargs):
            return self.estimator.predict_proba(*args,**kwargs)
        
        def get_params(self,*args,**kwargs):
            return self.estimator.get_params(*args,**kwargs);

        def get_coef_with_shap(self, X):

            if self.shap_coef_ is None:
                n_samples = X.shape[0]
                if X.shape[0] > 100: n_samples = 100;

                if estimator_type == 'tree':
                    explainer = shap.TreeExplainer(self.estimator,X,n_samples = n_samples)   
                else:
                    print('Initializing KernelExplainer, this might take a while... \n')
                    explainer = shap.KernelExplainer(self.estimator.predict_proba, X, n_samples = n_samples , link="logit")
                
                self.shap_values = explainer.shap_values(X)
                self.shap_coef_ = np.abs(self.shap_values).sum(axis=0)
            return self.shap_coef_

        @property
        def coef_(self):
            if self.use_shap:
                return self.get_coef_with_shap(self.X)
            else:
                return self.estimator.coef_