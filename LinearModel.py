import time
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prepareData import create_training_validation_test_data

# FNAME = 'rain_1h_3_0.8_2.1_EXTENDED'
# df = pd.read_csv("Input data/CustomData/"+FNAME+".csv", sep = ";")
# test = pd.DataFrame(df.groupby(by = ['edge', 'rain_1h_cat_3', 'dark']).agg({'Avg_Speed_adjusted':['min', 'mean', 'max']}))
# print(test)
"""------------------------  Load the data -------------------------"""
FNAME = 'rain_1h_3_0.8_2.1_0513'# 'rain_1h_3_0.8_2.1_EXTENDED'#'rain_1h_3_0.8_2.1_EXTENDED'#'rain_1h_3_0.8_2.1'#'rain_3h_3_2.5_6.3'#'rain_5h_3_4.2_10.4'#'rain_5h_3_2.5_7.5'#'rain_10h_3_8.3_20.8'
def image_converter(image_list_str):
    if image_list_str == "-": return image_list_str
    else:
        image_list = eval(image_list_str)
        return [np.array(image).astype(int) for image in image_list]
df = pd.read_csv("Input data/CustomData/"+FNAME+".csv", sep = ";", converters={'image': image_converter})
################################################################

# Filter out useless data (to be decided on) 
df = df[df.AmountObs >= 5].reset_index(drop = True)#print(df)
print(df)

# Create training/test data 
df_training, df_test_KR, df_test_UR, scaler = create_training_validation_test_data(FNAME = FNAME , df_satellite=df)
x_train_info, x_train_image, y_train, group_indices = df_training
"""-----------------------------------------------------------------"""

# Features INFO
# Midlon, Midlat, rain_1h_cat_3, dark,'swir_min', 'swir_max', 'swir_mean'
def testModel(model, df_test_KR, df_test_UR, scaler):
    x_test_info_KR, x_test_image_KR, y_test_KR = df_test_KR
    x_test_info_UR, x_test_image_UR, y_test_UR = df_test_UR
    return [x_test_info_KR, x_test_image_KR, y_test_KR.iloc[:,0], x_test_info_UR, x_test_image_UR, y_test_UR.iloc[:,0]]

x_test_info_KR, x_test_image_KR, y_test_KR, x_test_info_UR, x_test_image_UR, y_test_UR = testModel(None, df_test_KR, df_test_UR, None)

def pipeline(X_info, X_image, y, model, train=True, use_colors=True, use_info=True):
    X = X_info[:,0:4]
    
    if use_colors:
        # Add the average color of the sat image to the info used
        X = np.concatenate((X, X_image.mean(axis=(-2, -3))/255), axis=1) # three columns, one column per R, G, B
        X = np.concatenate((X, X_info[:,4:]), axis=1)
    
    if not use_info:
        # skip info columns
        X = X[:, 4:]
        #print(X.shape)
    
    print(X.shape)
    if train:
        model.fit(X, y)
    
    
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = mse**0.5
    print(f'mse: {mse:.2f}, rmse: {rmse:.2f}')


for kwargs in [dict(use_colors=True, use_info=True), dict(use_colors=False, use_info=True), dict(use_colors=True, use_info=False)]:
    print()
    print('-'*10 + f'Using options: {kwargs}' + '-'*10)
    lm = LinearRegression()
    print('-' * 10 + 'TRAINING FIT' + '-' * 10)
    pipeline(x_train_info, x_train_image, y_train, lm, **kwargs)
    print('-' * 10 + 'TEST FIT KNOWN ROADS' + '-' * 10)
    pipeline(x_test_info_KR, x_test_image_KR, y_test_KR, lm, train=False, **kwargs)
    print('-' * 10 + 'TEST FIT UNKNOWN ROADS' + '-' * 10)
    pipeline(x_test_info_UR, x_test_image_UR, y_test_UR, lm, train=False, **kwargs)
