import pickle
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler

from SETTINGS import *

# Set random seed
tf.random.set_seed(0)

def create_training_validation_test_data(FNAME, df_satellite):

    """
     Here, we divide the dataset in a training (+validation) and test set. We do this as follows: 
        - 10% of the ROADS will be used as unknown roads in the test set. So for these roads, no training
              was possible based on driving under other circumstances. 
        - 10% of the remaining data is then used as known roads in the test set. So these roads might be 
              trained based on driving on the same road under different circumstances. Eg, knowing that you can 
              drive rapidly during rainy weather probably tells you something about driving this road under dry 
              circumstances. 
     The rest is used as training set. We also store the indices corresponding to the roads in the training set 
     (group_indices). This allows us to also differentiate between known and unknown roads in the validation set
     (created when defining the deep learning structure).
    
     Input: df_roads 
     Output: x_train_info, x_train_image, y_train, group_indices
    """
    FEATURES = (ENV_FEATURES + INFO_FEATURES, IMAGE_FEATURE)
    PRINTING = True
    
    ################################################################
    ######################## Prepare the data ######################
    ################################################################
    if len(ENV_FEATURES + INFO_FEATURES)==0:
        FEATURES = (['midlon'], IMAGE_FEATURE)
    #df_satellite = df_satellite.loc[:,['edge', 'image_5', 'Avg_Speed_adjusted', 'AmountObs']+FEATURES[0]]
    
    a = list(df_satellite.columns)
    
    #df_satellite.sort_values(by=['edge', ENV_FEATURES[1], ENV_FEATURES[0]], inplace = True)
    df_satellite.sort_values(by=['edge'] + ENV_FEATURES, inplace = True)
    df_satellite.reset_index(drop=True, inplace = True)
    if PRINTING: 
        print("---------------------------------------------------------------------------------")
        print("------------------------------ PRE-PROCESSED DATA -------------------------------")
        print("---------------------------------------------------------------------------------")
        print(df_satellite)
        print("---------------------------------------------------------------------------------")
    ################################################################

    ################################################################
    #################### TEST / TRAIN INDICES ######################
    ################################################################
    # Define a random state to be able to reproduce results
    random_state = RANDOM_STATE

    # Define target variable
    y = df_satellite[[TARGET_VARIABLE, 'edge', 'AmountObs', 'image']+FEATURES[0]]

    # Define features
    features_info, feature_image = FEATURES[0], FEATURES[1]
    features = features_info + feature_image

    # Test set unknown roads
    test_UR = df_satellite.groupby(by=['edge']).agg({'midlon':'count'}).reset_index(drop=False)
    test_UR_edges = list(test_UR.sample(frac=TEST_SET_PERCENTAGE_UR, replace=False, random_state=random_state).edge)
    test_UR_indices = list(df_satellite[df_satellite.edge.isin(test_UR_edges)].index)
    # Test set known roads
    test_KR_indices = list(df_satellite[~df_satellite.edge.isin(test_UR_edges)].sample(frac=TEST_SET_PERCENTAGE_KR, replace=False, random_state=random_state).index)
    test_indices = test_UR_indices + test_KR_indices # indices that are used in the test set
    # Training set
    train_indices = list(df_satellite[~df_satellite.index.isin(test_indices)].index)
    # Set group indices training set (used for creating validaton datasets) 
    group_indices = df_satellite[df_satellite.index.isin(train_indices)].groupby(by=['edge']).indices.values()
    ################################################################

    ################################################################
    ###################### TEST / TRAIN SET ########################
    ################################################################
    # Define features
    df_satellite = df_satellite.loc[:,features]

    ######################### Training set #########################
    x_train = df_satellite[df_satellite.index.isin(train_indices)]
    x_train.reset_index(drop = True, inplace = True)
    y_train = y.loc[train_indices, TARGET_VARIABLE].reset_index(drop = True) # Target variable
    # Split the feature data in image data and info data
    x_train_info = x_train.loc[:, x_train.columns != 'image']
    print(x_train_info)
    # Scale all the feature variables (to make them equally important)]
    scaler = StandardScaler(with_std = True)
    # print("-----------------------------------------------------------------")
    # print(x_train_info)
    # print("-----------------------------------------------------------------")
    x_train_info = scaler.fit_transform(x_train_info).astype("float32")
    if PRINTING: print("----------------------------------------")
    if PRINTING: print("Dimensions of the training set (info):", x_train_info.shape)
    # Stack (create np array) from all the training images
    x_train_image = np.vstack([tuple(x_train.image.values)]).astype("float32")/255
    if PRINTING: print("Dimensions of the training set (images):", x_train_image.shape)
    if PRINTING: print("Dimension of (training) target variable", y_train.shape)
    if PRINTING: print("----------------------------------------")
    ################################################################
    
    ################### Test set (Unknown Roads) ###################
    x_test_UR = df_satellite[df_satellite.index.isin(test_UR_indices)]
    test_UR_indices = x_test_UR.index
    x_test_UR.reset_index(drop = True, inplace = True)
    y_test_UR = y.loc[test_UR_indices, [TARGET_VARIABLE, 'edge', 'AmountObs', 'image']+FEATURES[0]].reset_index(drop = True) # Target variable (test to numpy for ease when creating confusion matrix)
    # Split the feature data in image data and info data
    x_test_info_UR = x_test_UR.loc[:, x_test_UR.columns != 'image']
    x_test_info_UR = scaler.transform(x_test_info_UR).astype("float32")
    if PRINTING: print("----------------------------------------")
    if PRINTING: print("Dimensions of the \"Unknown Roads\" test set (info):", x_test_info_UR.shape)
    # Stack (create np array) from all the training images
    x_test_image_UR = np.vstack([tuple(x_test_UR.image.values)]).astype("float32")/255
    if PRINTING: print("Dimensions of the \"Unknown Roads\" test set (images):", x_test_image_UR.shape)
    if PRINTING: print("Dimension of \"Unknown Roads\" (test) target variable", y_test_UR.shape)
    if PRINTING: print("----------------------------------------")
    ################################################################
      
    #################### Test set (Known Roads) ####################
    x_test_KR = df_satellite[df_satellite.index.isin(test_KR_indices)]
    test_KR_indices = x_test_KR.index
    x_test_KR.reset_index(drop = True, inplace = True)
    y_test_KR = y.loc[test_KR_indices,[TARGET_VARIABLE, 'edge', 'AmountObs', 'image']+FEATURES[0]].reset_index(drop = True) # Target variable (test to numpy for ease when creating confusion matrix)
    # Split the feature data in image data and info data
    x_test_info_KR = x_test_KR.loc[:, x_test_KR.columns != 'image']
    x_test_info_KR = scaler.transform(x_test_info_KR).astype("float32")
    if PRINTING: print("----------------------------------------")
    if PRINTING: print("Dimensions of the \"Known Roads\" test set (info):", x_test_info_KR.shape)
    # Stack (create np array) from all the training images
    x_test_image_KR = np.vstack([tuple(x_test_KR.image.values)]).astype("float32")/255
    if PRINTING: print("Dimensions of the \"Known Roads\" test set (images):", x_test_image_KR.shape)
    if PRINTING: print("Dimension of \"Known Roads\" (test) target variable", y_test_KR.shape)
    if PRINTING: print("----------------------------------------")
    if PRINTING: print("---------------------------------------------------------------------------------")
    if PRINTING: print("---------------------------------------------------------------------------------")
    ################################################################
    
    return [x_train_info, x_train_image, y_train, group_indices], [x_test_info_KR, x_test_image_KR, y_test_KR], [x_test_info_UR, x_test_image_UR, y_test_UR], scaler

#create_training_validation_test_data()