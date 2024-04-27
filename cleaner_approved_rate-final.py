from azureml.core import Workspace, Dataset, Datastore
import numpy as np
import json
import pandas as pd
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
dataset_22 = Dataset.Tabular.from_delimited_files(path=(datastore, '2022_public_lar_csv.csv'))
dataset_21 = Dataset.Tabular.from_delimited_files(path=(datastore, '2021_public_lar.csv'))
dataset_20 = Dataset.Tabular.from_delimited_files(path=(datastore, '2020_lar_csv.csv'))
dataset_19 = Dataset.Tabular.from_delimited_files(path=(datastore, '2019_public_lar_csv.csv'))
dataset_18 = Dataset.Tabular.from_delimited_files(path=(datastore, '2018_public_lar_csv.csv'))
print('dataset complete')

lei_dataset = Dataset.Tabular.from_delimited_files(path=(datastore, 'lei_lookup.csv'))
df_22 = dataset_22.to_pandas_dataframe()
df_21 = dataset_21.to_pandas_dataframe()
df_20 = dataset_20.to_pandas_dataframe()
df_19 = dataset_19.to_pandas_dataframe()
df_18 = dataset_18.to_pandas_dataframe()

df_22 = df_22[['lei','state_code','derived_race','derived_sex','applicant_age_above_62','rate_spread','denial_reason_1','aus_1','debt_to_income_ratio', 'combined_loan_to_value_ratio','derived_loan_product_type','derived_dwelling_category','loan_term']]
df_21 = df_21[['lei','state_code','derived_race','derived_sex','applicant_age_above_62','rate_spread','denial_reason_1','aus_1','debt_to_income_ratio', 'combined_loan_to_value_ratio','derived_loan_product_type','derived_dwelling_category','loan_term']]
df_20 = df_20[['lei','state_code','derived_race','derived_sex','applicant_age_above_62','rate_spread','denial_reason_1','aus_1','debt_to_income_ratio', 'combined_loan_to_value_ratio','derived_loan_product_type','derived_dwelling_category','loan_term']]
df_19 = df_19[['lei','state_code','derived_race','derived_sex','applicant_age_above_62','rate_spread','denial_reason_1','aus_1','debt_to_income_ratio', 'combined_loan_to_value_ratio','derived_loan_product_type','derived_dwelling_category','loan_term']]
df_18 = df_18[['lei','state_code','derived_race','derived_sex','applicant_age_above_62','rate_spread','denial_reason_1','aus_1','debt_to_income_ratio', 'loan_to_value_ratio','derived_loan_product_type','derived_dwelling_category','loan_term']]]
df_18.rename(columns={"loan_to_value_ratio": "combined_loan_to_value_ratio"}, inplace = True)
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
df_cleaned = df.merge(lei_df, left_on='lei', right_on='LEI', how = 'left')
df_cleaned['Approved'] = np.where(df_cleaned['denial_reason_1'] == 10, True, False)
df_cleaned = df_cleaned[df_cleaned['derived_race'].isin(['Black or African American', 'White', 'Asian'])]
df_cleaned = df_cleaned[df_cleaned['derived_sex'].isin(['Male','Female'])]
df_cleaned = df_cleaned[df_cleaned['applicant_age_above_62'].isin(['Yes','No',True,False])]
df_cleaned = df_cleaned[df_cleaned['Approved']==True]
df_cleaned['Industry'] = df_cleaned['Entity.LegalName'].astype(str).apply(determine_industry)
df_cleaned = df_cleaned.replace('None', np.NaN)
cols_full = df_cleaned.columns.tolist()
num_cols = ['activity_year', 'derived_msa_md', 'county_code', 'census_tract', 'action_taken', 'purchaser_type', 'preapproval', 'loan_type', 'loan_purpose', 'lien_status', 'reverse_mortgage', 'open_end_line_of_credit', 'business_or_commercial_purpose', 'loan_amount', 'combined_loan_to_value_ratio', 'interest_rate', 'rate_spread', 'hoepa_status', 'total_loan_costs', 'total_points_and_fees', 'origination_charges', 'discount_points', 'lender_credits', 'loan_term', 'prepayment_penalty_term', 'intro_rate_period', 'negative_amortization', 'interest_only_payment', 'balloon_payment', 'other_nonamortizing_features', 'property_value', 'construction_method', 'occupancy_type', 'manufactured_home_secured_property_type', 'manufactured_home_land_property_interest', 'total_units', 'multifamily_affordable_units', 'income', 'applicant_credit_score_type', 'co_applicant_credit_score_type', 'submission_of_application', 'initially_payable_to_institution', 'aus_1', 'aus_2', 'aus_3', 'aus_4', 'aus_5', 'denial_reason_1', 'denial_reason_2', 'denial_reason_3', 'denial_reason_4', 'tract_population', 'tract_minority_population_percent', 'ffiec_msa_md_median_family_income', 'tract_to_msa_income_percentage', 'tract_owner_occupied_units', 'tract_one_to_four_family_homes', 'tract_median_age_of_housing_units', 'Approved', 'Industry']
cols = [x for x in cols_full if x not in num_cols]
print(cols)
for col in cols:
    print(col)
    entity_list = list(set(df_cleaned[col]))
    print(entity_list[0:3])
    length_col = list(range(0, len(entity_list)))
    inside_dict = {entity_list[i]:length_col[i] for i in range(len(entity_list))}
    df_cleaned[col] = df_cleaned[col].replace({col: inside_dict}, inplace = True)
    print(df_cleaned.head())
# entity_set = set(df_cleaned['Entity.LegalName'])
# entity_list = list(entity_set)
# length_col = list(range(0,len(entity_list)))
# entity_dict = dict(zip(entity_list, length_col))
# df_cleaned.replace({"Entity.LegalName": entity_dict}, inplace = True)
# ##clean states
# state_set = set(df_cleaned['state_code'])
# state_list = list(state_set)
# length_col = list(range(0,len(state_list)))
# state_dict = {state_list[i]:length_col[i] for i in range(len(state_list))}
# df_cleaned.replace({"state_code": state_dict}, inplace = True)
# ##clean race
# race_dict = {'Black or African American':0, 'White':1, 'Asian':2}
# df_cleaned.replace({"derived_race": race_dict}, inplace = True)
# ##clean sex
# sex_dict = {'Male':0,'Female':1}
# df_cleaned.replace({"derived_sex": sex_dict}, inplace = True)
# ##clean industry
# industry_dict = {'CU':0,'Bank':1,'Other':2}
# df_cleaned.replace({"Industry": industry_dict}, inplace = True)
# ##clean dti
# dti_set = set(df_cleaned['debt_to_income_ratio'])
# dti_list = list(dti_set)
# length_col = list(range(0,len(dti_list)))
# dti_dict = {dti_list[i]:length_col[i] for i in range(len(dti_list))}
# df_cleaned.replace({"debt_to_income_ratio": dti_dict}, inplace = True)
# ## clean age
# df_cleaned['applicant_age_above_62'] = df_cleaned['applicant_age_above_62'].astype(str)
# age_dict = {'True': 1, 'Yes': 1, 'False': 0, 'No': 0}
# df_cleaned['applicant_age_above_62'] = df_cleaned['applicant_age_above_62'].replace(age_dict)
# df_cleaned['applicant_age_above_62'] = df_cleaned['applicant_age_above_62'].astype(int)
# ## clean derived loan product type
# derived_loan_product_type_dict = {'Conventional:First Lien' : 1,
# 'FHA:First Lien' : 2,
# 'VA:First Lien' :3,
# 'FSA/RHS:First Lien':4,
# 'Conventional:Subordinate Lien':5,
# 'FHA:Subordinate Lien':6,
# 'VA:Subordinate Lien':7,
# 'FSA/RHS:Subordinate Lien':8}
# df_cleaned.replace({"derived_loan_product_type": derived_loan_product_type_dict}, inplace = True)
# ## clean derived dwelling category
# derived_dwelling_category_dict = {'Single Family (1-4 Units):Site-Built':0,
# 'Multifamily:Site-Built (5+ Units)':1,
# 'Single Family (1-4 Units):Manufactured':2,
# 'Multifamily:Manufactured (5+ Units)':3}
# df_cleaned.replace({"derived_dwelling_category": derived_dwelling_category_dict}, inplace = True)
df_cleaned = df_cleaned[['state_code','derived_race','derived_sex','applicant_age_above_62','rate_spread','aus_1','debt_to_income_ratio', 'combined_loan_to_value_ratio','derived_loan_product_type','derived_dwelling_category','loan_term','Entity.LegalName','Industry']]
print(df_cleaned.head())
df_cleaned.to_csv("df_cleaned_rate_all.csv", index = False, header = True)

