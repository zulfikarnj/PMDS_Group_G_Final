import pandas as pd
from pandas.api import types
from six import string_types
import re
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
import pycountry_convert as pc

def nullpercentage(df):
  """
  This is a function to check percentage of null data in each column of a
  pandas dataframe

  Parameters
  ----------------------
  df : pandas dataframe - default input dataframe

  Return
  ----------------------
  null_percent :  pandas dataframe - dataframe of null percentages 
  """
  null_percent = df.isnull().sum().sort_values(ascending = False)/len(df)*100
  print(null_percent) 

def rename_columns(df):
    """
    This is a function to rename feature names to more concise names
    Parameters
    ----------------------
    df : pandas dataframe - default input dataframe

    Return
    ----------------------
    df :  pandas dataframe - renamed feature name
    """

    col_names = {
    "Weekly Cases" : "WeekCase",
    "Weekly Cases per Million" : "WeekCasePerMil",
    "Weekly Deaths" : "WeekDeath",
    "Weekly Deaths per Million" : "WeekDeathPerMil",
    "Total Vaccinations" : "TotalVac",
    "People Vaccinated" : "PeopleVac",
    "People Fully Vaccinated" : "PeopleFullVac",
    "Total Boosters" : "TotalBoost",
    "Daily Vaccinations" : "DailyVac",
    "Total Vaccinations per Hundred" : "TotalVacPerHun",
    "People Vaccinated per Hundred" : "PeopleVacPerHun",
    "People Fully Vaccinated per Hundred" : "PeopleFullVacPerHun",
    "Total Boosters per Hundred" : "TotalBoostPerHun",
    "Daily Vaccinations per Hundred" : "DailyVacPerHun",
    "Daily People Vaccinated" : "DailyPeopleVac",
    "Daily People Vaccinated per Hundred" : "DailyPeopleVacPerHun",
    "Next Week's Deaths" : "NWD"}
    df = df.rename(columns = col_names)
    return df

def map_location(df_column):
    """
    This is a function to give label/number to a certain location

    input
    df_column : pandas dataframe column - data column with location information

    output
    df_column : pandas dataframe column - data column with coded location
    """
    location_map = {'EU' : "1",
                'AS' : "2",
                'AF' : "3",
                'OC' : "4",
                'SA' : "5",
                'World' : "6",
                'NA' : "7",
                'High income' : "8",
                'Low income' : "9",
                'Upper middle income' : "10",
                'Lower middle income' : "11"
                }
    df_column = df_column.map(location_map)
    return df_column

def map_year(row):
  """
    This is a function to map each year to a certain category

    input
    row : pandas dataframe row - data column with year information

    output
    drow : pandas dataframe row - data column with coded year
    """
  if row == 2022:
    row = "2"
  if row == 2020:
    row = "0"
  if row == 2021:
    row = "1"
  return row 

def decode_input(input_data, save_file=True, return_file=True):
    """
    Function to . 
        1. Drop fitur id
        2. Categorize country into continents


    Paramters
    ----------
    input_data      : pandas dataframe  - input dataframe
    save_file       : bool              - apabila True, akan menyimpan hasil data
    return_file     : bool              - apabila True, akan melakukan return

    Return
    -------
    output_data     : pandas dataframe  - output dataframe
    """
    # copy untuk hindari aliasing
    output_data = input_data.copy()

    # drop fitur id jika ada
    col = list(output_data.columns)
    if "Id" in col:
      output_data = output_data.drop(["Id"], axis=1)

    # Ubah data location menjadi per benua
    output_data['Location'] = output_data['Location'].apply(country_coder)

    if save_file:
        joblib.dump(output_data, "output_data_decode_v1.pickle", compress = 3)

    if return_file:
        return output_data

def country_coder(x):
    """
    This is a function to sort countries based on their continent.

    Parameters
    ---------------------
    x : str - default location
    
    Return
    ---------------------
    final_code : str - continent of the location
    """

    # Some countries found in the dataset are not found in the library and thus added manually 
    # in this function
    Africa_countries = ["Africa", "Democratic Republic of Congo", "Cote d'Ivoire"]
    European_countries = ["Europe", "European Union","Kosovo","Faeroe Islands"]
    Asian_countries = ["Asia", "Timor"]
    South_Am_countries = ['South America','Bonaire Sint Eustatius and Saba', 'Curacao']
    Oceania_countries = ['Oceania']
    North_Am_countries = ['North America']
    World_countries = ['International']
    try:  
      country_code = pc.country_name_to_country_alpha2(x, cn_name_format="default")
      final_code = pc.country_alpha2_to_continent_code(country_code)
    except (ValueError, RuntimeError, NameError, KeyError):
      final_code = x
    if final_code in Africa_countries:
      final_code = 'AF'
    if final_code in European_countries:
      final_code = 'EU'
    if final_code in Asian_countries:
      final_code = 'AS'
    if final_code in South_Am_countries:
      final_code = 'SA'
    if final_code in Oceania_countries:
      final_code = 'OC'
    if final_code in North_Am_countries:
      final_code = 'NA'
    if final_code in World_countries:
      final_code = 'World'    
    return final_code

def selectrows(df,kolom, place_list):
    """
    Function to select rows in dataframe based on column value

    Parameters
    ---------------
    df        : pandas dataframe  - input dataframe from which we would like to extract data
    kolom     : string            - our target column
    place_list: list              - list of values we would like to match

    Return
    ----------------
    df_result : pandas dataframe  - output dataframe
    """
    continents = ['EU', 'AF', 'OC', 'NA', 'AS', 'SA']
    income_class = ['Low income', 'Lower middle income', 'Upper middle income', 'High income']
    df_result = df.loc[df[kolom].isin(place_list)]

    return df_result

def map_incomeclass_and_continents(df, column_name, type,save_file=True, return_file=True):
    continents = ['EU', 'AF', 'OC', 'NA', 'AS', 'SA']
    income_class = ['Low income', 'Lower middle income', 'Upper middle income', 'High income']
    if type == 'continents':
        output = selectrows(df, column_name, continents)
        joblib.dump(output, "output/continent_v1.pickle", compress = 3)
    if type == 'income_class':
        output = selectrows(df, column_name, income_class)
        joblib.dump(output, "output/income_class_v1.pickle", compress = 3)
    
    if return_file:
        return output
    
    return output

def categorize_input(X_train, return_file=True):
    """
    Function to separate categorical to numerical input on input data. 
    The method is simple. If the column is string, the column will be sorted to 
    categorical column. If not, the column will be classified as numerical column.

    Parameters
    -----------
    X_train     : pandas dataframe  - Input training dataframe
    return_file : bool              - Apabila true, akan melakukan return data 

    Return
    -------
    numerical_columns   : list      - List dari numeric column/fitur
    categorical_columns : list      - List dari categorical column/fitur
    """
    # Ekstrak kolom
    df_columns = X_train.columns

    # Buat list penampung
    numerical_columns = []
    categorical_columns = []

    # Mencari
    for column in df_columns:
        if X_train[column].dtypes in ["int64", "float64"] :
            numerical_columns.append(column)
        else:
            categorical_columns.append(column)

    if return_file:
        return numerical_columns, categorical_columns

def split_input_output(dataset, target_column, save_file=True, return_file=True):
    """
    Function to separate dataset to input & output (based on target_column)

    Parameters
    -----------
    dataset         : pandas dataframe  - Dataset
    target_column   : str               - nama kolom yang jadi output
    save_file       : bool              - Apabila true, akan melakukan saving file dataframe dalam pickle
    return_file     : bool              - Apabila true, akan melakukan return data              
    
    Return
    -------
    input_df        : pandas dataframe  - dataframe input
    output_df       : pandas dataframe  - dataframe output
    """
    output_df = dataset[target_column]
    input_df = dataset.drop([target_column], axis=1)    # drop kolom target

    #Bagian dump ini bisa tidak diikutkan
    if save_file:
        joblib.dump(input_df, "output/input_df_v1.pickle", compress = 3)
        joblib.dump(output_df, "output/output_df_v1.pickle", compress = 3)

    if return_file:
        return input_df, output_df
    
def split_train_validation(input_df, output_df, save_file=True, return_file=True, test_size=0.2):
    """
    Fungsi untuk memisahkan dataset training menjadi training dataset & validation dataset
    untuk kebutuhan validasi, dengan perbandingan test_size = validation_dataset/total_dataset

    Parameters
    -----------
    input_df    : pandas dataframe  - dataframe input
    output_df   : pandas dataframe  - dataframe output
    save_file   : bool              - Apabila true, akan melakukan saving file dataframe dalam pickle
    return_file : bool              - Apabila true, akan melakukan return data  

    Return
    -------
    X_train           : pandas dataframe  - dataframe training input
    X_validation      : pandas dataframe  - dataframe validation input
    y_train           : pandas dataframe  - dataframe training output
    y_validation      : pandas dataframe  - dataframe validation output
    """
    # Copy data biar tidak terjadi aliasing
    X = input_df.copy()
    y = output_df.copy()

    # Split data
    # Random state = 123 untuk mempermudah duplikasi riset
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, 
                                                                    test_size=test_size,
                                                                    random_state=123)

    #Bagian dump ini bisa tidak diikutkan
    if save_file:
        joblib.dump(X_train, "output/X_train.pickle",compress = 3)
        joblib.dump(X_validation, "output/X_validation.pickle", compress = 3)
        joblib.dump(y_train, "output/y_train.pickle", compress = 3)
        joblib.dump(y_validation, "output/y_validation.pickle", compress = 3)

    if return_file:
        return X_train, X_validation, y_train, y_validation
    
def normalize_input(input_data):
    """
    Function to do normalize

    Parameters
    -----------
    input_data      : pandas dataframe  - input data which we want to standardize
    state           : str               - fitting or transformation process
    save_file       : bool              - if True, will save to new dataframe
    return_file     : bool              - if True, will do return

    Return
    -------
    output_data     : pandas dataframe  - standardisasi result dataframe
    """
    # Save column
    column_ = input_data.columns

    # Make scaler
    
    state = 'fit'
    scaler = StandardScaler()

    scaler.fit(input_data)
    normalized_data = scaler.transform(input_data)
    joblib.dump(scaler, "output/standard_scaler.pkl", compress = 3)
    return normalized_data
