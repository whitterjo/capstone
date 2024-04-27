from azureml.core import Workspace, Dataset, Datastore
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
import scipy.stats.distributions as dist
from io import StringIO

# subscription_id = ## redacted ##
resource_group = 'rg-Capstone'
workspace_name = 'stokerwj-capstone-ml'

workspace = Workspace(subscription_id, resource_group, workspace_name)
datastore = Datastore.get(workspace, "hmdadata")
print('datastore complete')
dataset = Dataset.Tabular.from_delimited_files(path=(datastore, 'hmdadatacleaned.csv'))
dataset_rate = Dataset.Tabular.from_delimited_files(path=(datastore, 'hmdadatacleanedapproved.csv'))
print('dataset complete')

df_cleaned = dataset.to_pandas_dataframe()
df_rate_cleaned = dataset_rate.to_pandas_dataframe()

age_dict = {'True': 'Yes', 'Yes': 'Yes', 'False': 'No', 'No': 'No'}

df_cleaned['applicant_age_above_62'] = df_cleaned['applicant_age_above_62'].astype(str)
df_rate_cleaned['applicant_age_above_62'] = df_cleaned['applicant_age_above_62'].astype(str)

df_cleaned['applicant_age_above_62'] = df_cleaned['applicant_age_above_62'].replace(age_dict)
df_rate_cleaned['applicant_age_above_62'] = df_rate_cleaned['applicant_age_above_62'].replace(age_dict)

results = []

def analyze_approval(group):
    # Calculate the counts of age group applicants
    age_count = group['applicant_age_above_62'].value_counts()
    count_above_62 = age_count.get('Yes', 0)
    count_below_62 = age_count.get('No', 0)
    above_62_approved = group[(group['applicant_age_above_62'] == 'Yes') & (group['Approved'] == 1)].shape[0]
    below_62_approved = group[(group['applicant_age_above_62'] == 'No') & (group['Approved'] == 1)].shape[0]
    above_62_denied = group[(group['applicant_age_above_62'] == 'Yes') & (group['Approved'] == 0)].shape[0]
    below_62_denied = group[(group['applicant_age_above_62'] == 'No') & (group['Approved'] == 0)].shape[0]

    try:
        # Chi-square test for approval rates
        contingency_table = group.pivot_table(index='Approved', columns='applicant_age_above_62', aggfunc='size', fill_value=0)
        chi2, p_approval, _, _ = chi2_contingency(contingency_table)
    except Exception:
        p_approval = np.nan

    # try:
    #     # Separate interest rates by age for t-test
    #     interest_rates_above_62 = group[group['applicant_age_above_62'] == 'Yes']['rate_spread']
    #     interest_rates_below_62 = group[group['applicant_age_above_62'] == 'No']['rate_spread']

    #     # t-test for interest rates
    #     if interest_rates_above_62.empty or interest_rates_below_62.empty:
    #         raise ValueError("One of the age groups has no data for interest rates.")
        
    #     t_stat, p_interest = ttest_ind(interest_rates_above_62, interest_rates_below_62, equal_var=False, nan_policy='omit')
    # except Exception:
    #     p_interest = np.nan

    # Return a Series including the counts, and p-values for approval rates and interest rates
    return pd.Series({'count_above_62': count_above_62, 'count_below_62': count_below_62, 'above_62_approved': above_62_approved, 'below_62_approved' : below_62_approved, 'above_62_denied': above_62_denied, 'below_62_denied' : below_62_denied, 'p_approval': p_approval})

def analyze_rate(group):
    # Calculate the counts of age group applicants
    # age_count = group['applicant_age_above_62'].value_counts()
    # count_above_62 = age_count.get('Yes', 0)
    # count_below_62 = age_count.get('No', 0)
    # above_62_approved = group[(group['applicant_age_above_62'] == 'Yes') & (group['Approved'] == 1)].shape[0]
    # below_62_approved = group[(group['applicant_age_above_62'] == 'No') & (group['Approved'] == 1)].shape[0]
    # above_62_denied = group[(group['applicant_age_above_62'] == 'Yes') & (group['Approved'] == 0)].shape[0]
    # below_62_denied = group[(group['applicant_age_above_62'] == 'No') & (group['Approved'] == 0)].shape[0]

    # try:
    #     # Chi-square test for approval rates
    #     contingency_table = group.pivot_table(index='Approved', columns='applicant_age_above_62', aggfunc='size', fill_value=0)
    #     chi2, p_approval, _, _ = chi2_contingency(contingency_table)
    # except Exception:
    #     p_approval = np.nan

    try:
        # Separate interest rates by age for t-test
        interest_rates_above_62 = group[group['applicant_age_above_62'] == 'Yes']['rate_spread']
        interest_rates_below_62 = group[group['applicant_age_above_62'] == 'No']['rate_spread']

        # t-test for interest rates
        if interest_rates_above_62.empty or interest_rates_below_62.empty:
            raise ValueError("One of the age groups has no data for interest rates.")
        
        t_stat, p_interest = ttest_ind(interest_rates_above_62, interest_rates_below_62, equal_var=False, nan_policy='omit')
    except Exception:
        p_interest = np.nan

    # Return a Series including the counts, and p-values for approval rates and interest rates
    return pd.Series({'p_interest': p_interest})

# Analysis at the institution and state level
detailed_results = df_cleaned.groupby(['Entity.LegalName', 'state_code']).apply(analyze_approval).reset_index()
detailed_results['aus_1'] = ''
detailed_results['Industry'] = ''
detailed_results['Year'] = ''
detailed_results_rate = df_rate_cleaned.groupby(['Entity.LegalName', 'state_code']).apply(analyze_rate).reset_index()
detailed_results_rate['aus_1'] = ''
detailed_results_rate['Industry'] = ''
detailed_results_rate['Year'] = ''
detailed_results = pd.merge(detailed_results,detailed_results_rate, how = 'left', on = ['Entity.LegalName', 'state_code','aus_1','Industry','Year'])
print('detailed results completed')

# Analysis at the state level only
state_level_results = df_cleaned.groupby('state_code').apply(analyze_approval).reset_index()
state_level_results['Entity.LegalName'] = ''  # Blank lei for state-level results
state_level_results['aus_1'] = ''
state_level_results['Industry'] = ''
state_level_results['Year'] = ''
state_level_rate_results = df_rate_cleaned.groupby('state_code').apply(analyze_rate).reset_index()
state_level_rate_results['Entity.LegalName'] = ''  # Blank lei for state-level results
state_level_rate_results['aus_1'] = ''
state_level_rate_results['Industry'] = ''
state_level_rate_results['Year'] = ''
state_level_results = pd.merge(state_level_results,state_level_rate_results, how = 'left', on = ['Entity.LegalName', 'state_code','aus_1','Industry','Year'])
print('state results completed')

# Analysis at the institution level only
lei_level_results = df_cleaned.groupby('Entity.LegalName').apply(analyze_approval).reset_index()
lei_level_results['state_code'] = ''  # Blank lei for state-level results
lei_level_results['aus_1'] = ''
lei_level_results['Industry'] = ''
lei_level_results['Year'] = ''
lei_level_results_rate = df_rate_cleaned.groupby('Entity.LegalName').apply(analyze_rate).reset_index()
lei_level_results_rate['state_code'] = ''  # Blank lei for state-level results
lei_level_results_rate['aus_1'] = ''
lei_level_results_rate['Industry'] = ''
lei_level_results_rate['Year'] = ''
lei_level_results = pd.merge(lei_level_results,lei_level_results_rate, how = 'left', on = ['Entity.LegalName', 'state_code','aus_1','Industry','Year'])
print('lei results completed')

#Analysis at the underwriting level only
aus_level_results = df_cleaned.groupby('aus_1').apply(analyze_approval).reset_index()
aus_level_results['state_code'] = ''  # Blank lei for state-level results
aus_level_results['Entity.LegalName'] = ''
aus_level_results['Industry'] = ''
aus_level_results['Year'] = ''
aus_level_results_rate = df_rate_cleaned.groupby('aus_1').apply(analyze_rate).reset_index()
aus_level_results_rate['state_code'] = ''  # Blank lei for state-level results
aus_level_results_rate['Entity.LegalName'] = ''
aus_level_results_rate['Industry'] = ''
aus_level_results_rate['Year'] = ''
aus_level_results = pd.merge(aus_level_results,aus_level_results_rate, how = 'left', on = ['Entity.LegalName', 'state_code','aus_1','Industry','Year'])
print('aus results completed')

#Analysis at the industry level only
industry_level_results = df_cleaned.groupby('Industry').apply(analyze_approval).reset_index()
industry_level_results['state_code'] = ''  # Blank lei for state-level results
industry_level_results['Entity.LegalName'] = ''
industry_level_results['aus_1'] = ''
industry_level_results['Year'] = ''
industry_level_results_rate = df_rate_cleaned.groupby('Industry').apply(analyze_rate).reset_index()
industry_level_results_rate['state_code'] = ''  # Blank lei for state-level results
industry_level_results_rate['Entity.LegalName'] = ''
industry_level_results_rate['aus_1'] = ''
industry_level_results_rate['Year'] = ''
industry_level_results = pd.merge(industry_level_results,industry_level_results_rate, how = 'left', on = ['Entity.LegalName', 'state_code','aus_1','Industry','Year'])

print('industry results completed')

#Analysis at the year level only
year_level_results = df_cleaned.groupby('Year').apply(analyze_approval).reset_index()
year_level_results['Industry'] = ''
year_level_results['state_code'] = ''  # Blank lei for state-level results
year_level_results['Entity.LegalName'] = ''
year_level_results['aus_1'] = ''
year_level_results_rate = df_rate_cleaned.groupby('Year').apply(analyze_rate).reset_index()
year_level_results_rate['Industry'] = ''
year_level_results_rate['state_code'] = ''  # Blank lei for state-level results
year_level_results_rate['Entity.LegalName'] = ''
year_level_results_rate['aus_1'] = ''
year_level_results = pd.merge(year_level_results,year_level_results_rate, how = 'left', on = ['Entity.LegalName', 'state_code','aus_1','Industry','Year'])
print('year results completed')

#State and industry
state_industry_results = df_cleaned.groupby(['Industry', 'state_code']).apply(analyze_approval).reset_index()
state_industry_results['Entity.LegalName'] = ''
state_industry_results['aus_1'] = ''
state_industry_results['Year'] = ''
state_industry_results_rate = df_rate_cleaned.groupby(['Industry', 'state_code']).apply(analyze_rate).reset_index()
state_industry_results_rate['Entity.LegalName'] = ''
state_industry_results_rate['aus_1'] = ''
state_industry_results_rate['Year'] = ''
state_industry_results = pd.merge(state_industry_results,state_industry_results_rate, how = 'left', on = ['Entity.LegalName', 'state_code','aus_1','Industry','Year'])
print('state and industry completed')

#Industry underwriting
industry_underwriting_results = df_cleaned.groupby(['Industry', 'aus_1']).apply(analyze_approval).reset_index()
industry_underwriting_results['state_code'] = ''  # Blank lei for state-level results
industry_underwriting_results['Entity.LegalName'] = ''
industry_underwriting_results['Year'] = ''
industry_underwriting_results_rate = df_rate_cleaned.groupby(['Industry', 'aus_1']).apply(analyze_rate).reset_index()
industry_underwriting_results_rate['state_code'] = ''  # Blank lei for state-level results
industry_underwriting_results_rate['Entity.LegalName'] = ''
industry_underwriting_results_rate['Year'] = ''
industry_underwriting_results = pd.merge(industry_underwriting_results,industry_underwriting_results_rate, how = 'left', on = ['Entity.LegalName', 'state_code','aus_1','Industry','Year'])
print('industry underwriting completed')

#Underwriting year
year_underwriting_results = df_cleaned.groupby(['Year', 'aus_1']).apply(analyze_approval).reset_index()
year_underwriting_results['state_code'] = ''  # Blank lei for state-level results
year_underwriting_results['Entity.LegalName'] = ''
year_underwriting_results['Industry'] = ''
year_underwriting_results_rate = df_rate_cleaned.groupby(['Year', 'aus_1']).apply(analyze_rate).reset_index()
year_underwriting_results_rate['state_code'] = ''  # Blank lei for state-level results
year_underwriting_results_rate['Entity.LegalName'] = ''
year_underwriting_results_rate['Industry'] = ''
year_underwriting_results = pd.merge(year_underwriting_results,year_underwriting_results_rate, how = 'left', on = ['Entity.LegalName', 'state_code','aus_1','Industry','Year'])
print('underwriting year completed')

#Institution year
institution_underwriting_results = df_cleaned.groupby(['Year', 'Entity.LegalName']).apply(analyze_approval).reset_index()
institution_underwriting_results['state_code'] = ''  # Blank lei for state-level results
institution_underwriting_results['aus_1'] = ''
institution_underwriting_results['Industry'] = ''
institution_underwriting_results_rate = df_rate_cleaned.groupby(['Year', 'Entity.LegalName']).apply(analyze_rate).reset_index()
institution_underwriting_results_rate['state_code'] = ''  # Blank lei for state-level results
institution_underwriting_results_rate['aus_1'] = ''
institution_underwriting_results_rate['Industry'] = ''
institution_underwriting_results = pd.merge(institution_underwriting_results,institution_underwriting_results_rate, how = 'left', on = ['Entity.LegalName', 'state_code','aus_1','Industry','Year'])
print('istitution year completed')

# Union the results
combined_results = pd.concat([detailed_results, state_level_results, lei_level_results, aus_level_results, industry_level_results, state_industry_results, industry_underwriting_results,year_level_results,year_underwriting_results,institution_underwriting_results], ignore_index=True).sort_values(by=['state_code', 'Entity.LegalName'])
combined_results = combined_results.loc[(combined_results['above_62_approved'] >= 10.0) & (combined_results['below_62_approved'] >= 10.0) & (combined_results['above_62_denied'] >= 10.0) & (combined_results['below_62_denied'] >= 10.0)]
combined_results['Approval 95% CI Interpretation'] = combined_results['p_approval'].apply(lambda p: "Reject" if p < 0.05 else "Accept")
combined_results['Interest Rate 95% CI Interpretation'] = combined_results['p_interest'].apply(lambda p: "Reject" if p < 0.05 else "Accept")
combined_results['Approval 99% CI Interpretation'] = combined_results['p_approval'].apply(lambda p: "Reject" if p < 0.01 else "Accept")
combined_results['Interest Rate 99% CI Interpretation'] = combined_results['p_interest'].apply(lambda p: "Reject" if p < 0.01 else "Accept")

# Optional: You might want to sort or reorganize the combined_results for better readability
combined_results.sort_values(['state_code', 'Entity.LegalName'], inplace=True, ascending=[True, True])

# Show the combined results
print(combined_results.head())

combined_results.to_csv("age_output.csv")
