from azureml.core import Workspace, Dataset, Datastore
import numpy as np
import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from scipy.stats import chi2_contingency
import xgboost as xgb

# subscription_id = ## redacted ##
resource_group = 'rg-Capstone'
workspace_name = 'stokerwj-capstone-ml'

workspace = Workspace(subscription_id, resource_group, workspace_name)
datastore = Datastore.get(workspace, "hmdadata")
print('datastore complete')
dataset = Dataset.Tabular.from_delimited_files(path=(datastore, 'hmdadatacleaned.csv'))

df = dataset.to_pandas_dataframe()

Crosstab = pd.crosstab(index = df_cleaned['aus_1'], columns = df_cleaned['derived_race'])
chi2, p_approval, _, _ = chi2_contingency(Crosstab)
print(Crosstab)
print(chi2)
print(p_approval)