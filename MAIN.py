import time
import pandas as pd
import random
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from multiprocessing import Pool
from tensorflow import keras

from prepareData import create_training_validation_test_data
from NeuralNets_0812 import mixed_neural_network_SPEED, mixed_neural_network_SPEED_onlySAT
from testModel import testModel, testModel_SAT

from SETTINGS import *
import os 

####################### Load the data ##########################
# FNAME = 'rain_1h_3_0.8_2.1_0513'#'rain_1h_3_0.8_2.1_EXTENDED'#'rain_1h_3_0.8_2.1'#'rain_3h_3_2.5_6.3'#'rain_5h_3_4.2_10.4'#'rain_5h_3_2.5_7.5'#'rain_10h_3_8.3_20.8'
def image_converter(image_list_str):
    if image_list_str == "-": return image_list_str
    else:
        image_list = eval(image_list_str)
        return [np.array(image).astype(int) for image in image_list]
df = pd.read_csv("Input data/CustomData/"+FNAME+".csv", sep = ";", converters={'image': image_converter})
################################################################

if GROUPBY_ENV:
    gb = ['rain_1h_cat_3', 'dark']
    for e in ['rain_1h_cat_3', 'dark']:
        if e not in ENV_FEATURES:
            gb.remove(e)
        
            aggregation = {
                TARGET_VARIABLE: lambda x: np.average(x, weights=df.loc[x.index, 'AmountObs']),
                'AmountObs': 'sum',
                **{col: 'first' for col in df.columns if col not in ['edge', e, TARGET_VARIABLE, 'AmountObs']+gb}
            }
            df = df.groupby(by=['edge']+gb).agg(aggregation).reset_index()

# Only use observations that are averaged over at least MIN_AMOUNT_OBS observations
df = df[df.AmountObs >= MIN_AMOUNT_OBS].reset_index(drop = True)

# remove features that are not used
possible_features = ['midlon', 'midlat', 'vegetation_percentage','swir_min', 'swir_max', 'swir_mean']
for f in possible_features:
    if f not in ENV_FEATURES + INFO_FEATURES:
        df[f] = 0

"""------------------- Create training/test data -------------------"""
df_training, df_test_KR, df_test_UR, scaler = create_training_validation_test_data(FNAME = FNAME , df_satellite=df)
x_train_info, x_train_image, y_train, group_indices = df_training
"""-----------------------------------------------------------------"""

"""-------------------------- Fit a model --------------------------"""
# Run the deep learning program for a specific set of settings and evaluate its performance
if CREATE_SINGLE_MODEL:
    if len(ENV_FEATURES + INFO_FEATURES)==0: mixed_neural_network_SPEED_onlySAT(x_train_info, x_train_image, y_train, group_indices, MODEL_SETTINGS, RANDOM_STATE, save_model = True, save_name = SAVE_NAME)
    else: mixed_neural_network_SPEED(x_train_info, x_train_image, y_train, group_indices, MODEL_SETTINGS, RANDOM_STATE, save_model = True, save_name = SAVE_NAME)
"""-----------------------------------------------------------------"""

"""------------------- Test a pre-computed model -------------------"""
if TEST_MODEL: 
    model = keras.models.load_model("Results/"+CASENAME+"/temp_best_models/"+CHECK_MODEL+'.h5', compile = False)
    if len(ENV_FEATURES + INFO_FEATURES)==0: testModel_SAT(model, df_test_KR, df_test_UR, scaler)
    else: testModel(model, df_test_KR, df_test_UR, scaler)
"""-----------------------------------------------------------------"""

if HYPER_PARAMETER_OPTIMIZATION:
    def do_something(x_train_info, x_train_image, y_train, selected_settings):
        global group_indices
        # EXtract settings
        ccd, cn, cmd, mn, md, lr, bs = selected_settings[0], selected_settings[1], selected_settings[2], selected_settings[3], selected_settings[4], selected_settings[5], selected_settings[6]
        mse, rmse, mae, huber, n_epochs = [], [],[], [], []
        for random_state in range(CROSS_VAL):
            mse_temp, rmse_temp, mae_temp, huber_temp, n_epochs_temp = mixed_neural_network_SPEED(x_train_info, x_train_image, y_train, group_indices = group_indices, hyper_settings=selected_settings, random_state = random_state, save_model = False)
            
            # Update metrics
            mse += [mse_temp]
            rmse += [rmse_temp]
            mae += [mae_temp]
            huber += [huber_temp]
            n_epochs += [n_epochs_temp]
        
        avg_mse = sum(mse)/CROSS_VAL
        avg_rmse = sum(rmse)/CROSS_VAL
        avg_mae = sum(mae)/CROSS_VAL
        avg_huber = sum(mse)/CROSS_VAL
        
        avg_ne = sum(n_epochs)/len(n_epochs)
        return list((ccd, cn, cmd, mn, md, lr, bs, avg_ne, avg_mse, avg_rmse, avg_mae, avg_huber))
    
    # # Define grid
    # PARAMETER_DICT = {'conv_dropout':[0.0, 0.25],#[0.1, 0.2, 0.3], #
    #                   'conv_mlp_layer_neuron': [320, 560, 800],#[320, 480, 640, 800],#
    #                   'conv_mlp_dropout': [0.5, 0.75], #[0.5, 0.6, 0.7, 0.8, 0.9], #
    #                   'combi_layer_neuron': [2**x for x in [12]], #[2**x for x in [4,5,6,7,8,9,10,11]],# TODO 6 ->
    #                   'combi_mlp_dropout':[0.25, 0.5, 0.75],#[0.1,0.3,0.5,0.7,0.9],#
    #                   'learning_rate': [0.001],#
    #                   'batchsize': [2**x for x in [7,8,9,10]]}#[2**x for x in [4,5,6,7,8,9]]} #

    """------------ Use sequential processing ------------"""
    #(tensorflow is already optimized to run on GPU. Credit: Cascha)
    start = time.time()
    result_df = pd.DataFrame(columns = list(PARAMETER_DICT.keys())+['Epochs', 'AVG_MSE', 'AVG_RMSE', 'AVG_MAE', 'AVG_HUBER'])
    ind = 0
    if not os.path.exists("Results/"+CASENAME+"/HyperparameterTuning"): os.makedirs("Results/"+CASENAME+"/HyperparameterTuning")
    for ccd in PARAMETER_DICT['conv_dropout']:
        for cn in PARAMETER_DICT['conv_mlp_layer_neuron']:
            for cmd in PARAMETER_DICT['conv_mlp_dropout']:
                for mn in PARAMETER_DICT['combi_layer_neuron']:
                    for md in PARAMETER_DICT['combi_mlp_dropout']:
                        for lr in PARAMETER_DICT['learning_rate']:         
                            for bs in PARAMETER_DICT['batchsize']:
                                selected_settings = [ccd,cn,cmd,mn,md,lr,bs]
                                setting_result = do_something(x_train_info, x_train_image, y_train, selected_settings)
                                result_df.loc[len(result_df.index)] = setting_result
                                if ind%1==0:result_df.to_csv("Results/"+CASENAME+"/HyperparameterTuning/"+SAVE_HP_RESULTS, sep = ";")
                                ind+=1
    result_df.to_csv("Results/"+CASENAME+"/HyperparameterTuning/"+SAVE_HP_RESULTS, sep = ";")
    print(time.time() - start, "seconds have passed. ")
    """---------------------------------------------------"""

    """--------------- Use multiprocessing ---------------"""
    # if __name__ == '__main__':
    #     start = time.time()
    #     with Pool(3) as p:  # 3 processes at a time
    #         reslist = []
    #         for ccd in PARAMETER_DICT['conv_dropout']:
    #             for cn in PARAMETER_DICT['conv_mlp_layer_neuron']:
    #                 for cmd in PARAMETER_DICT['conv_mlp_dropout']:
    #                     for mn in PARAMETER_DICT['combi_layer_neuron']:
    #                         for md in PARAMETER_DICT['combi_mlp_dropout']:
    #                             for lr in PARAMETER_DICT['learning_rate']:
    #                                 for bs in PARAMETER_DICT['batchsize']:
    #                                     selected_settings = [ccd,cn,cmd,mn,md,lr,bs]
    #
    #
    #                                     reslist += [p.apply_async(do_something, (x_train_info, x_train_image, y_train, selected_settings))]
    #
    #
    #         result_df = pd.DataFrame(columns = list(PARAMETER_DICT.keys())+['Epochs', 'MSE'])
    #         for result in reslist:
    #             result_df.loc[len(result_df.index)] = result.get()
    #             result_df.to_csv("result_df.csv", sep = ";")
    #         print(result_df)
    #
    #         result_df.to_csv("result_df.csv", sep = ";")
    #         print(time.time() - start, "seconds have passed. ")
    """---------------------------------------------------"""
