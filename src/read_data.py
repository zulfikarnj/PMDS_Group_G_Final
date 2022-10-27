import pandas as pd
from pandas.api import types
from six import string_types
import joblib

def read_data(train_path, test_path, save_file=True, return_file=True):
    """
    Function to open .csv files

    Parameters
    -----------
    path        : str   - Dataset path
    save_file   : bool  - If true, will save dataframe file in pickle
    return_file : bool  - If true, will do data return              
    
    Return
    -------
    data    : pandas dataframe  - dataframe from pandas environment
    """
    # Read data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    
    #Bagian dump ini bisa tidak diikutkan
    if save_file:
        joblib.dump(train, "output/train.pkl", compress = 3)
        joblib.dump(test, "output/test.pkl", compress = 3)
    
    if return_file:
        return train, test 
