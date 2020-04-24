import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def create_feature_importance_by_feature_weight(features,model):
    
    '''
        Create a Dataframe based on the Weights given to each feature by regression models, with the following columns:
            Weights: Weights of the models
            abs_Weights: Absolute value of the weights
            feature_importance: Normalized weights (all values of this column add up to one)
            abs_feature_importance: Absolute value of the feature importance
        
        Parameters:
            features: vector with the feature names
            model: trained model object. Must have the "coef_" parameter.
    '''

    df_weights = pd.DataFrame(index = features,columns = ['Weights'])
    df_weights['Weights'] = model.coef_.ravel()
    df_weights['abs_Weights'] = np.abs(df_weights['Weights'])
    total_weights = df_weights['abs_Weights'].values.ravel().sum()
    df_weights['feature_importance'] = df_weights['Weights'].values / total_weights
    df_weights['abs_feature_importance'] = df_weights['abs_Weights'].values / total_weights
    df_weights.sort_values(by=['abs_feature_importance'],ascending = False,inplace = True)
    return df_weights

def show_feature_importance_by_feature_weight(df_weights,model_title, color = None, absolute_values = False, feature_importance = False,figsize=(8,8)):

    '''
        Show a feature importance bar plot by model feature weights.

        Parameters:

        df_weights: Dataframe created by the function create_feature_importance_by_feature_weight(). 
        Contains the weights of the features and also their feature importances (values that add up to one)

        model_title: Title of the model to plot

        color: Tuple to give different colors to positive and negative weights. Example color = ('red','green')

        absolute_values: Flag to analyse just absolute values of weights

        feature_importance: Flag to select feature importance instead of weights 

    '''

    column = 'Weights'
    if absolute_values:
        column = 'abs_Weights'
    if feature_importance:
        column = column.replace('Weights','feature_importance')

    df_weights = df_weights.sort_values(by=[column],ascending = True,inplace = False)
    fig,ax = plt.subplots(1,figsize=figsize)
    colors = (0.2,0.4,0.8)
    if color is not None:
        color_mask = df_weights['feature_importance'] > 0
        colors = [color[0] if c else color[1] for c in color_mask]
        legend_elements = [
            Patch(facecolor = color[1], edgecolor='k', label='Negative Contribution'),
            Patch(facecolor = color[0], edgecolor='k', label='Positive Contribution')]
    
    ax.tick_params(axis = 'both',labelsize = 'large')
    df_weights[column].plot(kind = 'barh', grid = True, color = colors,edgecolor='k', alpha = 0.6,ax = ax)
    fig.suptitle(f'Feature Importance by Feature Weight - {model_title}',x = 0.3,fontsize = 20)

    if color:
        ax.legend(handles=legend_elements, bbox_to_anchor=(1, -0.05), borderaxespad=0.,fancybox=True, shadow=True,ncol = 2)

    if feature_importance:
        plt.figtext(0.91, 0.03, '*All absolute values add up to one', horizontalalignment='right')
    if absolute_values:
        plt.figtext(0.91, 0.01, '*All values are absolute', horizontalalignment='right')

    plt.subplots_adjust(top=0.93, bottom=0, left=0.10, right=0.95, hspace=0.40, wspace=0.35)
    plt.show()
    