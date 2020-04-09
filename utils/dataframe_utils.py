import pandas as pd

def show_column_options(df):
    print('Column Values:')
    cols = df.columns
    uniques = {}
    for col in cols:
        print(col,':',df[col].unique())
        uniques.update({col:df[col].unique()})
    return uniques