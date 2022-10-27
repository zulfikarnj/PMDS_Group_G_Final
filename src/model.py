import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import sklearn
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
import yaml
import time

def select_model(train_log_dict):
    temp = []
    for score in train_log_dict['model_score']:
        temp.append(score['rmse'])
    #print(temp)
    best_model = train_log_dict['model_name'][temp.index(min(temp))]
    
    
    return best_model

def read_preprocessed(params):
    x_train = joblib.load(params['X_PATH_TRAIN'])
    y_train = joblib.load(params['Y_PATH_TRAIN'])
    x_valid = joblib.load(params['X_PATH_VALID'])
    y_valid = joblib.load(params['Y_PATH_VALID'])
    x_test = joblib.load(params['X_PATH_TEST'])

    return x_train, y_train, x_valid, y_valid, x_test

def GBT_model(x, y):
    GBT_fit = GradientBoostingRegressor(min_samples_leaf = 4,
                                     min_samples_split = 10,
                                     max_features = 7,
                                     max_depth = 49,
                                     n_estimators = 500,
                                     subsample = 0.2,
                                     random_state = 0)
    GBT_fit.fit(x,y)
    return GBT_fit

def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    mape = metrics.mean_absolute_percentage_error(true, predicted)
    exp_var = metrics.explained_variance_score(true, predicted)
    return mae, mse, rmse, r2_square, mape, exp_var

def GBTfit(x_train, y_train):
    """
    Fit model

    Args:
        - model(callable): Sklearn / imblearn model
        - model_param(dict): sklearn's RandomizedSearchCV param_distribution
        - general_params(dict):x general parameters for the function
            - target(str) : y column to be used   
            - scoring(str) : sklearn cross-val scoring scheme
            - n_iter_search : RandomizedSearchCV number of iteration
    """
    #print( general_params['scoring'])

    model_fitted = GBT_model(x_train, y_train)
    
    return model_fitted

def validation_score(x_valid, y_valid, model_fitted):
    
    # Report default
    y_predicted = model_fitted.predict(x_valid)
    mae, mse, rmse, r2_square, mape, exp_var = evaluate(y_valid, y_predicted)
    score = {'mae':mae, 'mse':mse, 'rmse':rmse, 'r2': r2_square, 'mape': mape, 'exp_var': exp_var}
    print(rmse)
    return score

def main(params):

    #lasso = model_lasso
    #rf = model_rf
    #lsvr = model_svr

    train_log_dict = {'model_name': [],
                        'fit_time': [],
                        'model_score' : []}
   

    x_train, y_train, x_valid, y_valid, x_test  = read_preprocessed(params)
    t0 = time.time()
    train_log_dict['model_name'].append('Gradient_boost')
    fitted_model = GBTfit(x_train, y_train)
    elapsed_time = time.time() - t0
    print(f'elapsed time: {elapsed_time} s \n')
    train_log_dict['fit_time'].append(elapsed_time)
    score = validation_score( x_valid, y_valid, fitted_model)
    train_log_dict['model_score'].append(score)

    best_model = select_model(
        train_log_dict)
    print(
        f"Model: {best_model}")
    y_predicted = fitted_model.predict(x_test)
    
    joblib.dump(fitted_model,'output/model_used.pickle', compress = 3)
    return y_predicted
    #joblib.dump(best_parameter, 'output/isrelated_parameter.pkl')
    #joblib.dump(train_log_dict, 'output/isrelated_train_log.pkl')