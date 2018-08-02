import pandas as pd


dataset_challenge = pd.read_csv('/Users/pranjalg/work/ai-newbies/AI Innovation /challenge_set_uniformColumnNames.csv',low_memory=False)

print(len(dataset_challenge))
dataset_challenge.dropna(subset=['renewed_yorn'],  inplace = True)
print(len(dataset_challenge))


import numpy as np
final_features = ['innovation_challenge_key', 'contract_line_multi_year_discount_yorn', 'low_dollar_contract_yorn',
                  'hardware_yorn', 'minor_line_yorn', 'component_type',
                  'contract_line_repair_service_order_level', 'product_route_to_market', 'service_route_to_market_code',
                  'software_yorn', 'scms_name', 'product_category',
                  'product_setup_classification', 'item_status', 'software_support_services_eligible_yorn',
                  'product_classification', 'four_quarter_bookings_band', 'global_customer_market_segment_name',
                  'global_top_vertical_market_name', 'status_code', 'partner_tier',
                  'service_partner_base_partner_tier', 'warranty_type', 'product_type', 'detail_vertical_market_name',
                  "sales_node_renewal_rate", 'customer_renewal_rate', 'product_renewal_rate',
                  'service_sales_node_installed_base_sales_node_renewal_rate',
                  'service_partner_installed_base_partner_renewal_rate',
                  'service_fee_amount', 'mapped_service_list_price', 'contract_line_net_usd_amount',
                  'product_net_price','renewed_yorn']

numerical_cols = ["sales_node_renewal_rate", 'customer_renewal_rate', 'product_renewal_rate',
                  'service_sales_node_installed_base_sales_node_renewal_rate',
                  'service_partner_installed_base_partner_renewal_rate']

other_numerical_cols = ['service_fee_amount', 'mapped_service_list_price', 'contract_line_net_usd_amount',
                        'product_net_price']

categorical_cols = ['contract_line_multi_year_discount_yorn', 'low_dollar_contract_yorn',
                    'hardware_yorn', 'component_type',
                    'contract_line_repair_service_order_level', 'product_route_to_market',
                    'service_route_to_market_code',
                    'software_yorn', 'scms_name', 'product_category',
                    'product_setup_classification', 'item_status', 'software_support_services_eligible_yorn',
                    'product_classification', 'four_quarter_bookings_band', 'global_customer_market_segment_name',
                    'global_top_vertical_market_name', 'status_code', 'partner_tier',
                    'service_partner_base_partner_tier', 'warranty_type', 'product_type', 'detail_vertical_market_name']


dataset_challenge = dataset_challenge[dataset_challenge['minor_line_yorn'] == 'N']

#g = dataset_challenge[dataset_challenge['renewed_yorn'] == 'N']

#dataset = pd.concat([dataset_challenge,g])
#dataset_challenge = dataset
y = dataset_challenge['renewed_yorn'].values





from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

dataset_challenge = dataset_challenge[final_features]

dataset_challenge = dataset_challenge.drop(['renewed_yorn'], axis = 1)






for cols in categorical_cols:
    dataset_challenge[cols] = dataset_challenge[cols].fillna("NULL_VALUES")

for cols in numerical_cols:
    dataset_challenge[cols] = dataset_challenge[cols].fillna(dataset_challenge[cols].median())

for cols in other_numerical_cols:
    dataset_challenge[cols] = dataset_challenge[cols].fillna(dataset_challenge[cols].median())




for i in categorical_cols:
    le = LabelEncoder()
    dataset_challenge[i] = le.fit_transform(dataset_challenge[i])

data_encoded_chal = dataset_challenge.iloc[:,1:]

data_encoded_chal = data_encoded_chal.drop(['minor_line_yorn'], axis = 1)

#dataset_training = pd.get_dummies(data_encoded_chal, columns = categorical_cols, drop_first=True)
dataset_training = data_encoded_chal
X = dataset_training.values


sample_weight = np.random.RandomState(42).rand(y.shape[0])



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(X, y, sample_weight, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train1 = sc.fit_transform(X_train)
X_test1 = sc.transform(X_test)


import pprint
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 10)]
# Number of features to consider at every split
max_features = ['log2', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(20,160, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2,4,6,8, 10, 12,14,16,18]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 3,4,5,6,7,8]
# Method of selecting samples for training each tree
bootstrap = [False,True]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap
              }
pprint.pprint(random_grid)


import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [True],
    'max_depth': [30,32,38,40],
    'max_features': ['sqrt'],
    'min_samples_leaf': [1,2,3,4,5],
    'min_samples_split': [2,3,4],
    'n_estimators': [322,300,280,340]
}


# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                          cv = 5, n_jobs = -1, verbose = 2, scoring='log_loss')

grid_search.fit(X_train1, y_train)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
print("training")
rf = RandomForestClassifier(n_jobs=-1)
# Random search of parameters, using 10 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5,verbose=2, random_state=0,n_jobs = -1, scoring="neg_log_loss", iid=False)
# Fit the random search model
rf_random.fit(X_train1, y_train)


from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
classifier = RandomForestClassifier(bootstrap =  False,
 max_depth = 32,
 max_features = 'log2',
 min_samples_leaf = 1,
 min_samples_split = 4,
 n_estimators = 888,
class_weight="balanced")
clf_isotonic = CalibratedClassifierCV(clf, cv=2, method='isotonic')
clf_isotonic.fit(X_train1, y_train, sw_train)
y_pred_prob = clf_isotonic.predict_proba(X_test1)
y_pred = clf_isotonic.predict(X_test1)