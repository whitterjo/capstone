from azureml.core import Workspace, Dataset, Datastore
import numpy as np
import json
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import f1_score

def determine_industry(legal_name):
    if 'credit union' in legal_name.lower():
        return 'CU'
    elif 'bank' in legal_name.lower():
        return 'Bank'
    else:
        return 'Other'

# subscription_id = ## redacted ##
resource_group = 'rg-Capstone'
workspace_name = 'stokerwj-capstone-ml'

workspace = Workspace(subscription_id, resource_group, workspace_name)
datastore = Datastore.get(workspace, "hmdadata")
print('datastore complete')
dataset = Dataset.Tabular.from_delimited_files(path=(datastore, 'hmdadatacleaned.csv'))
print('dataset complete')

df = dataset.to_pandas_dataframe()

## cleaning
df['rate_spread'] = df['rate_spread'].replace("NA", np.nan)
df['rate_spread'] = df['rate_spread'].replace("Exempt", np.nan)
df['rate_spread'] = df_22['rate_spread'].astype(float)
df['combined_loan_to_value_ratio'] = df['combined_loan_to_value_ratio'].replace("NA", np.nan)
df['combined_loan_to_value_ratio'] = df['combined_loan_to_value_ratio'].replace("Exempt", np.nan)
df['combined_loan_to_value_ratio'] = df['combined_loan_to_value_ratio'].astype(float)

print('dataframe complete')
print(df.head())
df['debt_to_income_ratio'] = df['debt_to_income_ratio'].replace("NA", np.nan)
df['debt_to_income_ratio'] = df['debt_to_income_ratio'].replace("Exempt", np.nan)
df = df[['applicant_age_above_62','debt_to_income_ratio']]
df = df[df['applicant_age_above_62'].isin([True,False])]
df['applicant_age_above_62'] = df['applicant_age_above_62'].astype(str)
df.dropna(inplace = True)

table = df.pivot_table(index = 'debt_to_income_ratio',columns = 'applicant_age_above_62', aggfunc = 'size', fill_value = 0)
chi2, p_approval, _, _ = chi2_contingency(table)
print(table)
print(chi2)
print(p_approval)

table.to_csv('age_dti_pivot.csv')