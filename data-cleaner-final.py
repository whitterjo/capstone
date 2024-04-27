from azureml.core import Workspace, Dataset, Datastore
import numpy as np
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import f1_score
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import io
import os

## subscription_id = '## redacted ##
resource_group = 'rg-Capstone'
workspace_name = 'stokerwj-capstone-ml'

workspace = Workspace(subscription_id, resource_group, workspace_name)
datastore = Datastore.get(workspace, "hmdadata")
print('datastore complete')
dataset_22 = Dataset.Tabular.from_delimited_files(path=(datastore, '2021_public_lar.csv'))
dataset_21 = Dataset.Tabular.from_delimited_files(path=(datastore, '2021_public_lar.csv'))
dataset_20 = Dataset.Tabular.from_delimited_files(path=(datastore, '2020_lar_csv.csv'))
dataset_19 = Dataset.Tabular.from_delimited_files(path=(datastore, '2019_public_lar_csv.csv'))
dataset_18 = Dataset.Tabular.from_delimited_files(path=(datastore, '2018_public_lar_csv.csv'))
print('dataset complete')

def determine_industry(legal_name):
    if 'credit union' in legal_name.lower():
        return 'CU'
    elif 'bank' in legal_name.lower():
        return 'Bank'
    else:
        return 'Other'

lei_dataset = Dataset.Tabular.from_delimited_files(path=(datastore, 'lei_lookup.csv'))
df_22 = dataset_22.to_pandas_dataframe()
df_21 = dataset_21.to_pandas_dataframe()
df_20 = dataset_20.to_pandas_dataframe()
df_19 = dataset_19.to_pandas_dataframe()
df_18 = dataset_18.to_pandas_dataframe()

df_22 = df_22[['lei','state_code','derived_race','derived_sex','applicant_age_above_62','denial_reason_1','aus_1','debt_to_income_ratio', 'combined_loan_to_value_ratio']]
df_21 = df_21[['lei','state_code','derived_race','derived_sex','applicant_age_above_62','rate_spread','denial_reason_1','aus_1','debt_to_income_ratio', 'combined_loan_to_value_ratio']]
df_20 = df_20[['lei','state_code','derived_race','derived_sex','applicant_age_above_62','rate_spread','denial_reason_1','aus_1','debt_to_income_ratio', 'combined_loan_to_value_ratio']]
df_19 = df_19[['lei','state_code','derived_race','derived_sex','applicant_age_above_62','rate_spread','denial_reason_1','aus_1','debt_to_income_ratio', 'combined_loan_to_value_ratio']]
df_22 = df_22[['lei','state_code','derived_race','derived_sex','applicant_age_above_62','rate_spread','denial_reason_1','aus_1','debt_to_income_ratio', 'loan_to_value_ratio']]
df_22.rename(columns={"loan_to_value_ratio": "combined_loan_to_value_ratio"}, inplace = True)
lei_df = lei_dataset.to_pandas_dataframe()
lei_df = lei_df[['LEI', 'Entity.LegalName']]
## cleaning 2022
df_22['rate_spread'] = df_22['rate_spread'].replace("NA", np.nan)
df_22['rate_spread'] = df_22['rate_spread'].replace("Exempt", np.nan)
df_22['rate_spread'] = df_22['rate_spread'].astype(float)
df_22['Year'] = '2022'
df_22['combined_loan_to_value_ratio'] = df_22['combined_loan_to_value_ratio'].replace("NA", np.nan)
df_22['combined_loan_to_value_ratio'] = df_22['combined_loan_to_value_ratio'].replace("Exempt", np.nan)
df_22['combined_loan_to_value_ratio'] = df_22['combined_loan_to_value_ratio'].astype(float)
df_22['applicant_age_above_62'] = df_22['applicant_age_above_62'].replace("NA", np.nan)
df_22['applicant_age_above_62'] = df_22['applicant_age_above_62'].replace("Exempt", np.nan)

## cleaning 2021
df_21['rate_spread'] = df_21['rate_spread'].replace("NA", np.nan)
df_21['rate_spread'] = df_21['rate_spread'].replace("Exempt", np.nan)
df_21['rate_spread'] = df_21['rate_spread'].astype(float)
df_21['Year'] = '2021'
df_21['combined_loan_to_value_ratio'] = df_21['combined_loan_to_value_ratio'].replace("NA", np.nan)
df_21['combined_loan_to_value_ratio'] = df_21['combined_loan_to_value_ratio'].replace("Exempt", np.nan)
df_21['combined_loan_to_value_ratio'] = df_21['combined_loan_to_value_ratio'].astype(float)

## cleaning 2020
df_20['rate_spread'] = df_20['rate_spread'].replace("NA", np.nan)
df_20['rate_spread'] = df_20['rate_spread'].replace("Exempt", np.nan)
df_20['rate_spread'] = df_20['rate_spread'].astype(float)
df_20['Year'] = '2020'
df_20['combined_loan_to_value_ratio'] = df_20['combined_loan_to_value_ratio'].replace("NA", np.nan)
df_20['combined_loan_to_value_ratio'] = df_20['combined_loan_to_value_ratio'].replace("Exempt", np.nan)
df_20['combined_loan_to_value_ratio'] = df_20['combined_loan_to_value_ratio'].astype(float)

##cleaning 2019
df_19['rate_spread'] = df_19['rate_spread'].replace("NA", np.nan)
df_19['rate_spread'] = df_19['rate_spread'].replace("Exempt", np.nan)
df_19['rate_spread'] = df_19['rate_spread'].astype(float)
df_19['Year'] = '2019'
df_19['combined_loan_to_value_ratio'] = df_19['combined_loan_to_value_ratio'].replace("NA", np.nan)
df_19['combined_loan_to_value_ratio'] = df_19['combined_loan_to_value_ratio'].replace("Exempt", np.nan)
df_19['combined_loan_to_value_ratio'] = df_19['combined_loan_to_value_ratio'].astype(float)

##cleaning 2018
df_18['rate_spread'] = df_18['rate_spread'].replace("NA", np.nan)
df_18['rate_spread'] = df_18['rate_spread'].replace("Exempt", np.nan)
df_18['rate_spread'] = df_18['rate_spread'].astype(float)
df_18['Year'] = '2018'
df_18['combined_loan_to_value_ratio'] = df_18['combined_loan_to_value_ratio'].replace("NA", np.nan)
df_18['combined_loan_to_value_ratio'] = df_18['combined_loan_to_value_ratio'].replace("Exempt", np.nan)
df_18['combined_loan_to_value_ratio'] = df_18['combined_loan_to_value_ratio'].astype(float)

print('dataframe complete')

df = pd.concat([df_21, df_22, df_20, df_19, df_18], ignore_index=True)
df = df.merge(lei_df, left_on='lei', right_on='LEI', how = 'left')
df['Approved'] = np.where(df['denial_reason_1'] == 10, True, False)
df = df[['Entity.LegalName','state_code','derived_race','derived_sex','applicant_age_above_62','Approved','aus_1','debt_to_income_ratio', 'combined_loan_to_value_ratio', 'Year']]
df_cleaned = df[df['derived_race'].isin(['Black or African American', 'White', 'Asian'])]
df_cleaned = df_cleaned[df_cleaned['derived_sex'].isin(['Male','Female'])]
df_cleaned = df_cleaned[df_cleaned['applicant_age_above_62'].isin(['Yes','No',True,False])]
df_cleaned['Industry'] = df_cleaned['Entity.LegalName'].astype(str).apply(determine_industry)
df_cleaned = df_cleaned[['Entity.LegalName','state_code','derived_race','derived_sex','applicant_age_above_62','aus_1','Industry','debt_to_income_ratio','combined_loan_to_value_ratio','Year','Approved']]
df_cleaned.dropna(inplace=True)
print(df_cleaned.head())
print(df_cleaned.shape)
df_cleaned.to_csv("hmdadatacleaned.csv")