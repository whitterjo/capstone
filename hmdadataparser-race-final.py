from azureml.core import Workspace, Dataset, Datastore
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
import scipy.stats.distributions as dist
from scipy import stats
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

print(df_cleaned.head())
print(df_cleaned.shape)

results = []

def analyze_approval(group):
    # Calculate the counts of white and black applicants
    race_count = group['derived_race'].value_counts()
    count_white = race_count.get('White', 0)
    count_black = race_count.get('Black or African American', 0)
    white_approved = group[(group['derived_race'] == 'White') & (group['Approved'] == 1)].shape[0]
    black_approved = group[(group['derived_race'] == 'Black or African American') & (group['Approved'] == 1)].shape[0]
    asian_approved = group[(group['derived_race'] == 'Asian') & (group['Approved'] == 1)].shape[0]
    white_denied = group[(group['derived_race'] == 'White') & (group['Approved'] == 0)].shape[0]
    black_denied = group[(group['derived_race'] == 'Black or African American') & (group['Approved'] == 0)].shape[0]
    asian_denied = group[(group['derived_race'] == 'Asian') & (group['Approved'] == 0)].shape[0]


    try:
        # Chi-square test for approval rates
        contingency_table = group.pivot_table(index='Approved', columns='derived_race', aggfunc='size', fill_value=0)
        chi2, p_approval, _, _ = chi2_contingency(contingency_table)
    except Exception:
        p_approval = np.nan

    # # try:
    #     # Separate interest rates by race for t-test
    # interest_rates_white = group[group['derived_race'] == 'White']['rate_spread']
    # interest_rates_white = interest_rates_white[~np.isnan(interest_rates_white)]
    # interest_rates_black = group[group['derived_race'] == 'Black or African American']['rate_spread']
    # interest_rates_black = interest_rates_black[~np.isnan(interest_rates_black)]
    # interest_rates_asian = group[group['derived_race'] == 'Asian']['rate_spread']
    # interest_rates_asian = interest_rates_asian[~np.isnan(interest_rates_asian)]

    #     # # t-test for interest rates
    #     # if interest_rates_white.empty or interest_rates_black.empty or interest_rates_asian.empty:
    #     #     raise ValueError("One of the race groups has no data for interest rates.")
        
    #     # else:
    # F, p_interest = stats.f_oneway(interest_rates_white, interest_rates_black, interest_rates_asian)
    # except Exception:
    #     p_interest = np.nan

    # Return a Series including the counts, and p-values for approval rates and interest rates
    return pd.Series({'count_white': count_white, 'count_black': count_black, 'approved_white': white_approved, 'approved_black' : black_approved, 'approved_asian' : asian_approved, 'denied_white': white_denied, 'denied_black' : black_denied, 'denied_asian': asian_denied, 'p_approval': p_approval})

def analyze_rate(group):
    # Calculate the counts of white and black applicants
    # race_count = group['derived_race'].value_counts()
    # count_white = race_count.get('White', 0)
    # count_black = race_count.get('Black or African American', 0)
    # white_approved = group[(group['derived_race'] == 'White') & (group['Approved'] == 1)].shape[0]
    # black_approved = group[(group['derived_race'] == 'Black or African American') & (group['Approved'] == 1)].shape[0]
    # asian_approved = group[(group['derived_race'] == 'Asian') & (group['Approved'] == 1)].shape[0]
    # white_denied = group[(group['derived_race'] == 'White') & (group['Approved'] == 0)].shape[0]
    # black_denied = group[(group['derived_race'] == 'Black or African American') & (group['Approved'] == 0)].shape[0]
    # asian_denied = group[(group['derived_race'] == 'Asian') & (group['Approved'] == 0)].shape[0]


    # try:
    #     # Chi-square test for approval rates
    #     contingency_table = group.pivot_table(index='Approved', columns='derived_race', aggfunc='size', fill_value=0)
    #     chi2, p_approval, _, _ = chi2_contingency(contingency_table)
    # except Exception:
    #     p_approval = np.nan

    # try:
        # Separate interest rates by race for t-test
    interest_rates_white = group[group['derived_race'] == 'White']['rate_spread']
    interest_rates_white = interest_rates_white[~np.isnan(interest_rates_white)]
    interest_rates_black = group[group['derived_race'] == 'Black or African American']['rate_spread']
    interest_rates_black = interest_rates_black[~np.isnan(interest_rates_black)]
    interest_rates_asian = group[group['derived_race'] == 'Asian']['rate_spread']
    interest_rates_asian = interest_rates_asian[~np.isnan(interest_rates_asian)]

        # # t-test for interest rates
        # if interest_rates_white.empty or interest_rates_black.empty or interest_rates_asian.empty:
        #     raise ValueError("One of the race groups has no data for interest rates.")
        
        # else:
    F, p_interest = stats.f_oneway(interest_rates_white, interest_rates_black, interest_rates_asian)
    # except Exception:
    #     p_interest = np.nan

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
combined_results = combined_results.loc[(combined_results['approved_white'] >= 10.0) & (combined_results['approved_black'] >= 10.0) & (combined_results['denied_white'] >= 10.0) & (combined_results['denied_black'] >= 10.0) & (combined_results['denied_asian'] >= 10.0) & (combined_results['approved_asian'] >= 10.0)]
combined_results['Approval 95% CI Interpretation'] = combined_results['p_approval'].apply(lambda p: "Reject" if p < 0.05 else "Accept")
combined_results['Interest Rate 95% CI Interpretation'] = combined_results['p_interest'].apply(lambda p: "Reject" if p < 0.05 else "Accept")
combined_results['Approval 99% CI Interpretation'] = combined_results['p_approval'].apply(lambda p: "Reject" if p < 0.01 else "Accept")
combined_results['Interest Rate 99% CI Interpretation'] = combined_results['p_interest'].apply(lambda p: "Reject" if p < 0.01 else "Accept")

# Sorting results
combined_results.sort_values(['state_code', 'Entity.LegalName'], inplace=True, ascending=[True, True])

# Show the combined results
print(combined_results.head())

combined_results.to_csv("race_output.csv")
