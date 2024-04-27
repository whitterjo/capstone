from azureml.core import Workspace, Dataset, Datastore
import numpy as np
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import f1_score

subscription_id = '1ad0fa3f-7440-4513-bf5a-992b677c6f92'
resource_group = 'rg-Capstone'
workspace_name = 'stokerwj-capstone-ml'

workspace = Workspace(subscription_id, resource_group, workspace_name)
datastore = Datastore.get(workspace, "hmdadata")
print('datastore complete')
dataset_22 = Dataset.Tabular.from_delimited_files(path=(datastore, 'hmdadatacleanedapproved18.csv'))
print('dataset complete')

# def determine_industry(legal_name):
#     if 'credit union' in legal_name.lower():
#         return 'CU'
#     elif 'bank' in legal_name.lower():
#         return 'Bank'
#     else:
#         return 'Other'

# lei_dataset = Dataset.Tabular.from_delimited_files(path=(datastore, 'lei_lookup.csv'))
df_cleaned = dataset_22.to_pandas_dataframe()
# # df_21 = dataset_21.to_pandas_dataframe()
# # df_20 = dataset_20.to_pandas_dataframe()
# # df_19 = dataset_19.to_pandas_dataframe()
# # df_18 = dataset_18.to_pandas_dataframe()

# # df_22 = df_22[['lei','state_code','derived_race','derived_sex','applicant_age_above_62','rate_spread','denial_reason_1','aus_1','debt_to_income_ratio', 'combined_loan_to_value_ratio']]
# # df_21 = df_21[['lei','state_code','derived_race','derived_sex','applicant_age_above_62','rate_spread','denial_reason_1','aus_1','debt_to_income_ratio', 'combined_loan_to_value_ratio']]
# # df_20 = df_20[['lei','state_code','derived_race','derived_sex','applicant_age_above_62','rate_spread','denial_reason_1','aus_1','debt_to_income_ratio', 'combined_loan_to_value_ratio']]
# # df_19 = df_19[['lei','state_code','derived_race','derived_sex','applicant_age_above_62','rate_spread','denial_reason_1','aus_1','debt_to_income_ratio', 'combined_loan_to_value_ratio']]
# # df_18 = df_18[['lei','state_code','derived_race','derived_sex','applicant_age_above_62','rate_spread','denial_reason_1','aus_1','debt_to_income_ratio', 'loan_to_value_ratio']]
# # df_18.rename(columns={"loan_to_value_ratio": "combined_loan_to_value_ratio"}, inplace = True)
# # lei_df = lei_dataset.to_pandas_dataframe()
# # lei_df = lei_df[['LEI', 'Entity.LegalName']]
# # ## cleaning 2022
# # df_22['rate_spread'] = df_22['rate_spread'].replace("NA", np.nan)
# # df_22['rate_spread'] = df_22['rate_spread'].replace("Exempt", np.nan)
# # df_22['rate_spread'] = df_22['rate_spread'].astype(float)
# # df_22['Year'] = '2022'

# # df_22['combined_loan_to_value_ratio'] = df_22['combined_loan_to_value_ratio'].replace("NA", np.nan)
# # df_22['combined_loan_to_value_ratio'] = df_22['combined_loan_to_value_ratio'].replace("Exempt", np.nan)
# # df_22['combined_loan_to_value_ratio'] = df_22['combined_loan_to_value_ratio'].astype(float)
# # ## cleaning 2021
# # df_21['rate_spread'] = df_21['rate_spread'].replace("NA", np.nan)
# # df_21['rate_spread'] = df_21['rate_spread'].replace("Exempt", np.nan)
# # df_21['rate_spread'] = df_21['rate_spread'].astype(float)
# # df_21['Year'] = '2021'

# # df_21['combined_loan_to_value_ratio'] = df_21['combined_loan_to_value_ratio'].replace("NA", np.nan)
# # df_21['combined_loan_to_value_ratio'] = df_21['combined_loan_to_value_ratio'].replace("Exempt", np.nan)
# # df_21['combined_loan_to_value_ratio'] = df_21['combined_loan_to_value_ratio'].astype(float)
# # ## cleaning 2020
# # df_20['rate_spread'] = df_20['rate_spread'].replace("NA", np.nan)
# # df_20['rate_spread'] = df_20['rate_spread'].replace("Exempt", np.nan)
# # df_20['rate_spread'] = df_20['rate_spread'].astype(float)
# # df_20['Year'] = '2020'

# # df_20['combined_loan_to_value_ratio'] = df_20['combined_loan_to_value_ratio'].replace("NA", np.nan)
# # df_20['combined_loan_to_value_ratio'] = df_20['combined_loan_to_value_ratio'].replace("Exempt", np.nan)
# # df_20['combined_loan_to_value_ratio'] = df_20['combined_loan_to_value_ratio'].astype(float)
# # ##cleaning 2019
# # df_19['rate_spread'] = df_19['rate_spread'].replace("NA", np.nan)
# # df_19['rate_spread'] = df_19['rate_spread'].replace("Exempt", np.nan)
# # df_19['rate_spread'] = df_19['rate_spread'].astype(float)
# # df_19['Year'] = '2019'

# # df_19['combined_loan_to_value_ratio'] = df_19['combined_loan_to_value_ratio'].replace("NA", np.nan)
# # df_19['combined_loan_to_value_ratio'] = df_19['combined_loan_to_value_ratio'].replace("Exempt", np.nan)
# # df_19['combined_loan_to_value_ratio'] = df_19['combined_loan_to_value_ratio'].astype(float)
# # ##cleaning 2018
# # df_18['rate_spread'] = df_18['rate_spread'].replace("NA", np.nan)
# # df_18['rate_spread'] = df_18['rate_spread'].replace("Exempt", np.nan)
# # df_18['rate_spread'] = df_18['rate_spread'].astype(float)
# # df_18['Year'] = '2018'

# # df_18['combined_loan_to_value_ratio'] = df_18['combined_loan_to_value_ratio'].replace("NA", np.nan)
# # df_18['combined_loan_to_value_ratio'] = df_18['combined_loan_to_value_ratio'].replace("Exempt", np.nan)
# # df_18['combined_loan_to_value_ratio'] = df_18['combined_loan_to_value_ratio'].astype(float)

# # print('dataframe complete')

# # df = pd.concat([df_21, df_22, df_20, df_19, df_18], ignore_index=True)
# # df = df.merge(lei_df, left_on='lei', right_on='LEI', how = 'left')
# # df['Approved'] = np.where(df['denial_reason_1'] == 10, True, False)
# # df = df[['Entity.LegalName','state_code','derived_race','derived_sex','applicant_age_above_62','rate_spread','Approved','aus_1','debt_to_income_ratio', 'combined_loan_to_value_ratio', 'Year']]
# # df_cleaned = df[df['derived_race'].isin(['Black or African American', 'White', 'Asian'])]
# # df_cleaned = df_cleaned[df_cleaned['derived_sex'].isin(['Male','Female'])]
# # df_cleaned = df_cleaned[df_cleaned['applicant_age_above_62'].isin(['Yes','No'])]
# # df_cleaned['Industry'] = df_cleaned['Entity.LegalName'].astype(str).apply(determine_industry)
# df_cleaned = df_cleaned[['Entity.LegalName','state_code','derived_race','derived_sex','applicant_age_above_62','Approved','aus_1','Industry','debt_to_income_ratio','combined_loan_to_value_ratio','Year']]
# print(df_cleaned.head())
# cols = ['Entity.LegalName','state_code','derived_race','derived_sex','applicant_age_above_62','Approved','aus_1','Industry','debt_to_income_ratio','combined_loan_to_value_ratio','Year']
##clean age
df_cleaned['applicant_age_above_62'] = df_cleaned['applicant_age_above_62'].astype(str)
age_dict = {'True': 1, 'Yes': 1, 'False': 0, 'No': 0}
df_cleaned['applicant_age_above_62'] = df_cleaned['applicant_age_above_62'].replace(age_dict)
df_cleaned['applicant_age_above_62'] = df_cleaned['applicant_age_above_62'].astype(int)
# age_dict = {True: 1, 'Yes': 1, False: 0, 'No': 0}
# df_cleaned['applicant_age_above_62'] = df_cleaned['applicant_age_above_62'].replace(age_dict)
# age_dict = {'No':0,'Yes':1}
# df_cleaned.replace({"applicant_age_above_62": age_dict}, inplace = True)

## clean entities
entity_set = set(df_cleaned['Entity.LegalName'])
entity_list = list(entity_set)
length_col = list(range(0,len(entity_list)))
entity_dict = {entity_list[i]:length_col[i] for i in range(len(entity_list))}
df_cleaned.replace({"Entity.LegalName": entity_dict}, inplace = True)
##clean states
state_set = set(df_cleaned['state_code'])
state_list = list(state_set)
length_col = list(range(0,len(state_list)))
state_dict = {state_list[i]:length_col[i] for i in range(len(state_list))}
df_cleaned.replace({"state_code": state_dict}, inplace = True)
##clean race
race_dict = {'Black or African American':0, 'White':1, 'Asian':2}
df_cleaned.replace({"derived_race": race_dict}, inplace = True)
##clean sex
sex_dict = {'Male':0,'Female':1}
df_cleaned.replace({"derived_sex": sex_dict}, inplace = True)
##clean industry
industry_dict = {'CU':0,'Bank':1,'Other':2}
df_cleaned.replace({"Industry": industry_dict}, inplace = True)
##clean dti
dti_set = set(df_cleaned['debt_to_income_ratio'])
dti_list = list(dti_set)
length_col = list(range(0,len(dti_list)))
dti_dict = {dti_list[i]:length_col[i] for i in range(len(dti_list))}
df_cleaned.replace({"debt_to_income_ratio": dti_dict}, inplace = True)
derived_loan_product_type_dict = {'Conventional:First Lien' : 1,
'FHA:First Lien' : 2,
'VA:First Lien' :3,
'FSA/RHS:First Lien':4,
'Conventional:Subordinate Lien':5,
'FHA:Subordinate Lien':6,
'VA:Subordinate Lien':7,
'FSA/RHS:Subordinate Lien':8}
df_cleaned.replace({"derived_loan_product_type": derived_loan_product_type_dict}, inplace = True)
# ##clean approved
# approved_dict = {False:0, True:1}
# df_cleaned.replace({"Approved": approved_dict}, inplace = True)
# print(df_cleaned.head())
df_cleaned.to_csv("numeric_cleanedaproved18.csv")
# for col in cols:
#     inside_col = set(df_cleaned[col])
#     inside_col_2 = list(inside_col)
#     length_col = list(range(0, len(inside_col)))
#     inside_dict = {inside_col_2[i]:length_col[i] for i in range(len(inside_col_2))}
#     df_cleaned = df_cleaned.replace({col: inside_dict}, inplace = True)
# df_cleaned = df_cleaned[['Entity.LegalName','state_code','derived_race','derived_sex','applicant_age_above_62','aus_1','Industry','debt_to_income_ratio', 'combined_loan_to_value_ratio','Year','Approved']]
# df_cleaned.dropna(inplace=True)
# # df_cleaned = df_cleaned.sample(n=10000)
# X = df_cleaned.drop(['Approved'],1)
# y = df_cleaned['Approved']
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)
# n_estimators = [int(x) for x in np.linspace(start = 50, stop = 150, num = 10)]
# max_features = ['auto', 'sqrt']
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth = [2,4]
# max_depth.append(None)
# min_samples_split = [2, 5, 10]
# min_samples_leaf = [1, 2, 4]
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}

# rf = RandomForestClassifier()
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# rf_random.fit(X_train,y_train)
# best_params = rf_random.best_params_
# print(best_params)
# #param_df = pd.DataFrame.from_dict(best_params)
# #param_df.to_csv("RF_best_params.csv")
# rf_best = RandomForestClassifier(**best_params)
# #y_pred = rf_best.predict(X_test)
# # rf_classifier = RandomForestClassifier(rf_best)
# with open('best_params.txt', 'w') as convert_file: 
#      convert_file.write(json.dumps(best_params))
# rf_best = rf_best.fit(X_train,y_train)
# y_pred = rf_best.predict(X_test)
# cm = metrics.confusion_matrix(y_test,y_pred)
# con_mat_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])
# f1 = f1_score(y_test, y_pred)
# print(f1)
# feature_imp_df = pd.DataFrame(rf_best.feature_importances_, index = X.columns, columns = ['Feature Score'])
# feature_imp_df.sort_values(by = 'Feature Score', ascending = False, inplace = True)
# print(feature_imp_df)
# feature_imp_df.to_csv("Feature_Importance_RF_tree.csv")
