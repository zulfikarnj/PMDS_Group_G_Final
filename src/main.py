import numpy as np
import pandas as pd
import joblib
import yaml
from read_data import read_data
from preprocess import nullpercentage, rename_columns, map_location
from preprocess import map_year, decode_input, country_coder
from preprocess import selectrows, map_incomeclass_and_continents
from preprocess import categorize_input, split_input_output
from preprocess import split_train_validation, normalize_input
from model import read_preprocessed, GBT_model, evaluate, GBTfit
from model import validation_score, main, select_model  

#Open yaml
f = open("src/params/param.yaml", "r")
params = yaml.load(f, Loader=yaml.SafeLoader)
f.close()

#Open data
train_data,test_data = read_data(params['train_path'], params['test_path'])

#Preprocess data
train_data = train_data.fillna(0)
test_data = test_data.fillna(0)

train_data = rename_columns(train_data)
test_data = rename_columns(test_data)

train_coded_location = decode_input(train_data,return_file = True)
test_coded_location = decode_input(test_data,return_file = True)

train_coded_location['Year'] = train_coded_location['Year'].apply(map_year)
test_coded_location['Year'] = test_coded_location['Year'].apply(map_year)

train_coded_location['Location'] = map_location(train_coded_location['Location'])
test_coded_location['Location'] = map_location(test_coded_location['Location'])

label_column = params['target_column']
test_size = params['test_size']

train_model = train_coded_location
test_model = test_coded_location

input_df, output_df = split_input_output(train_model, label_column)

X_train, X_valid, y_train, y_valid = split_train_validation(input_df,
                                                    output_df,
                                                    test_size)

X_test = test_coded_location
joblib.dump(X_test, "output/X_test.pickle", compress = 3)

numerical_columns, categorical_columns = categorize_input(X_train)
X_train[numerical_columns] = normalize_input(X_train[numerical_columns])
X_test[numerical_columns] = normalize_input(X_test[numerical_columns])
X_valid[numerical_columns] = normalize_input(X_valid[numerical_columns])

joblib.dump(X_train, "output/X_train.pickle", compress = 3)
joblib.dump(X_valid, "output/X_validation.pickle", compress = 3)
joblib.dump(X_test, "output/X_test.pickle", compress = 3)

# Predict
x_train, y_train, x_valid, y_valid, x_test = read_preprocessed(params)
y_predicted = main(params)
print(y_predicted)