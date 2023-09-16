"""-------------------------------------------------------------------------"""
"""---------------------- PARAMETERS OF THE ALGORITHM ----------------------"""
"""-------------------------------------------------------------------------"""
CASENAME = 'FINAL_NEW_BEST'
RANDOM_STATE = 0
#__________________________________ Data ___________________________________#
FNAME = 'rain_1h_3_0.8_2.1_0513'
MIN_AMOUNT_OBS = 5
#___________________________________________________________________________#

#_________________________ Information that we use _________________________#
TARGET_VARIABLE = 'Avg_Speed_adjusted'
ENV_FEATURES = ['rain_1h_cat_3', 'dark']  # []
GROUPBY_ENV = False
INFO_FEATURES = ['midlon', 'midlat', 'swir_min', 'swir_max', 'swir_mean']  # ['midlon', 'midlat','swir_min', 'swir_max', 'swir_mean']
IMAGE_FEATURE = ['image']  # ['image']
#___________________________________________________________________________#

#________________________ Create training/test data ________________________#
TEST_SET_PERCENTAGE_UR = 0.1 # UNKNOWN ROADS
TEST_SET_PERCENTAGE_KR = 0.1 # KNOWN ROADS
VALIDATION_SPLIT = 0.2
#___________________________________________________________________________#

#__________________________ Create a single model __________________________#
CREATE_SINGLE_MODEL = True
if CREATE_SINGLE_MODEL:
    SAVE_NAME = 'test'
    
    ###################### NEW BEST MODEL #####################
    # MSE:    [0.25, 320, 0.5, 4096, 0.5, 0.001, 128]
    # MSE_2:  [0.25, 560, 0.5, 4096, 0.25, 0.003, 256]
    # MSE_NORAIN: [0.25, 560, 0.75, 4096, 0.25, 0.002, 256]
    # MSE_NODARK: [0.25, 320, 0.75, 4096, 0.25, 0.001, 256]
    # MSE_NOLOCATION: [0.25, 800, 0.75, 4096, 0.75, 0.002, 256]
    # MSE_NODARKNORAIN: [0.25, 320, 0.75, 4096, 0.75, 0.001, 256]
    # MSE_ONLYSAT: [0.25, 560, 0.75, 4096, 0.25, 0.003, 512]
    # MSE_ONLYSATIMAGE: [0.25, 800, 0.75, 4096, 0.5, 0.002, 256]
    ############################################################
    MODEL_SETTINGS = [0.25, 560, 0.75, 4096, 0.25, 0.003, 512]
    LOSS = 'mse' #mse, mae
    HUBER_DELTA = 100
    #_____________________________ Plot figure _____________________________#
    PLOT_LOSS_CURVES = False
    PLOT_FEATURE_MAPS = False
    if PLOT_FEATURE_MAPS: img = 3
    #_______________________________________________________________________#
#___________________________________________________________________________#

#___________________________ Test a single model ___________________________#
TEST_MODEL = True
if TEST_MODEL:
    CHECK_MODEL = "test"
    #_______________________________________________________________________#
    EXAMINE_SPEED_DISTRIBUTION = False
    #_______________________________________________________________________#
    EXAMINE_NUM_OBSERVATIONS = False
    #_______________________________________________________________________#
    CREATE_ROAD_NETWORK_PREDICTION = False
    if CREATE_ROAD_NETWORK_PREDICTION:
        FULL_SET = True
        if FULL_SET: 
            PATH_TO_DF_FULL = "Input data/CustomData/df_roads_FULL.csv"
    #_______________________________________________________________________#
    CREATE_HEATMAP_SPEED = False # Require df_test_all (CREATE_ROAD_NETWORK_PREDICTION)
    if CREATE_HEATMAP_SPEED:
        CUSTOM_SPEED_CATS = False
        if CUSTOM_SPEED_CATS: 
            SPEED_CATS = [15,30]
        if not CUSTOM_SPEED_CATS:
            SPEED_CATS = 3
    #_______________________________________________________________________#
    EXAMINE_PREDICTION_SPREAD = False # Require df_test_all (CREATE_ROAD_NETWORK_PREDICTION)
    #_______________________________________________________________________#
    CREATE_ENHANCED_ROAD_NETWORK = False # Require df_test_all (CREATE_ROAD_NETWORK_PREDICTION)
    if CREATE_ENHANCED_ROAD_NETWORK:
        PATH_TO_GRAPH = "Input data/Data/_Graphs_SIMPLE/"
    #_______________________________________________________________________#
#___________________________________________________________________________#

#________________________ Hyperparameter optimization ______________________#
HYPER_PARAMETER_OPTIMIZATION = False
if HYPER_PARAMETER_OPTIMIZATION:
    
    LOSS = 'mae' #mse, mae    
    HUBER_DELTA = 100
    PLOT_LOSS_CURVES = False
    PLOT_FEATURE_MAPS = False
    
    CROSS_VAL = 3
    
    PARAMETER_DICT = {'conv_dropout':[0.0, 0.25],#[0.1, 0.2, 0.3], #
                      'conv_mlp_layer_neuron': [320, 560, 800],#[320, 480, 640, 800],# 
                      'conv_mlp_dropout': [0.5, 0.75], #[0.5, 0.6, 0.7, 0.8, 0.9], #
                      'combi_layer_neuron': [2**x for x in [6,7,8,9,10,11,12]], #[2**x for x in [4,5,6,7,8,9,10,11]],# TODO 6 -> 
                      'combi_mlp_dropout':[0.25, 0.5, 0.75],#[0.1,0.3,0.5,0.7,0.9],#
                      'learning_rate': [0.001],#
                      'batchsize': [2**x for x in [7,8,9,10]]}#[2**x for x in [4,5,6,7,8,9]]} #
    SAVE_HP_RESULTS = "result_df_"+LOSS+".csv"
#___________________________________________________________________________#
"""-------------------------------------------------------------------------"""

    # MSE:    [0.25, 800, 0.50, 2048, 0.25, 0.003, 256]
    # MSE_2:  [0.25, 560, 0.75, 4096, 0.25, 0.002, 256]
    # MSE_NORAIN: [0.25, 560, 0.5, 4096, 0.25, 0.002, 256]
    # MSE_NODARK: [0.25, 560, 0.75, 4096, 0.75, 0.002, 256]
    # MSE_NOLOCATION: [0.25, 560, 0.75, 2048, 0.25, 0.002, 256]
    # MSE_NODARKNORAIN: [0.25, 320, 0.5, 4096, 0.5, 0.002, 256]
    # MSE_ONLYSAT: [0.25, 560, 0.75, 4096, 0.25, 0.001, 256]
    # MSE_ONLYSATIMAGE: [0.25, 800, 0.75, 4096, 0.5, 0.002, 256]
    # MAE:    [0.25, 800, 0.75, 4096, 0.50, 0.002, 256]
    # HUBER:  [0.25, 320, 0.75, 4096, 0.50, 0.001, 256]