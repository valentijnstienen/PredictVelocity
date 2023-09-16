import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import os
import networkx as nx
import math

from shapely import wkt
from math import radians, cos, sin, asin, sqrt, atan2, pi

""" --------------------------------------------------------------------------------------- """
""" ------------------------------ CREATE SPECIFIC DATAFRAMES ----------------------------- """
""" --------------------------------------------------------------------------------------- """
""" -------- Settings ------- """
CREATE_ROAD_DF = True
if CREATE_ROAD_DF:
    NUM_SPEED_CATS = 3
CREATE_WEATHER_DF = False
CREATE_RAIN_DF = False
if CREATE_RAIN_DF:
    AMOUNT_RAIN_CAT = 3
CREATE_FINAL_COMPARE = False
""" ------------------------- """
    
# Load the master dataframe (ONLY USED WHEN CREATING THE DATAFRAMES)
if CREATE_ROAD_DF | CREATE_WEATHER_DF | CREATE_RAIN_DF | CREATE_FINAL_COMPARE:
    # Define a custom converter function that extracts the image from the list
    def geom_converter(wkt_str):
        return wkt.loads(wkt_str)
    df = pd.read_table("Data/df_prepared_3.csv", sep = ";", index_col = 0, converters={'geometry':geom_converter}, low_memory=False)
    print(df)
    
    # def image_converter(image_list_str):
    #     if image_list_str == "-": return image_list_str
    #     else:
    #         image_list = eval(image_list_str)
    #         return [np.array(image).astype(int) for image in image_list]
    # def geom_converter(wkt_str):
    #     return wkt.loads(wkt_str)
    # df = pd.read_table("Data/df_prepared_3.csv", sep = ";", index_col = 0, converters={'geometry':geom_converter, 'image': image_converter}, low_memory=False)#, 'image': image_converter
    
# AVAILABLE FEATURES
# ['u', 'v', 'key', 'osmid', 'highway', 'maxspeed', 'oneway', 'new', 'geometry', 'length', 'DateDriven', 'DateDriven_r', 'Speed', 'temp',
# 'visibility', 'dew_point', 'feels_like', 'temp_min', 'temp_max', 'pressure', 'sea_level', 'grnd_level', 'humidity', 'wind_speed',
# 'wind_deg', 'wind_gust', 'rain_1h', 'rain_3h', 'rain_5h', 'rain_10h', 'rain_24h', 'snow_1h', 'snow_3h', 'clouds_all', 'weather_main',
# 'weather_description', 'image', 'cloud_cover', 'vegetation_percentage', 'swir_min', 'swir_max', 'swir_mean', 'nimages', 'generation_time']
############################################################################################
#################################### CREATE ROAD DF ########################################
############################################################################################
"""
 Here, we group the data by ['edge', 'dark', 'rain_5h_cat_3']. The latter two indicators 
 define a circumstance in which a vehicle has to drive. Note that, at this stage, we do 
 include as much other features as possible.

 Input : df_prepared_3
 Output : df_roads
"""
def plotImages(images, labels = None):
    """
     This procedure can be used to visualize a specific satellite image,
     using the df_satellite dataset (that contains the images themselves)
     a maximum of 40 images can be plotted at the same time
    """
    # Plot max 40 images
    number_figures = len(images)
    
    # Determine widht,height
    max_d = 5
    w = int(min(number_figures, max_d))
    h = int(1 + np.floor((number_figures-1) / max_d))
    try: a = images.index[0]
    except: a = 0
    plt.figure()
    for i in range(0,min(len(images), 40)):
        plt.subplot(h,w,i+1)
        plt.xticks([])
        plt.yticks([])
        if labels is not None:
            plt.title(str(labels[i]))#df_satellite.highway[i])
        plt.grid(False)
        maxValue = np.amax(eval(images[i+a]))
        minValue = np.amin(eval(images[i+a]))
        plt.imshow(eval(images[i+a]))
    plt.show()
if CREATE_ROAD_DF:
    """------------- Settings -------------"""
    ENV_VARIABLES = ['rain_1h', 'dark', 'vegetation_percentage', 'swir_min', 'swir_max', 'swir_mean'] # rain_5h_cat_3
    RAIN_CATS = [0.8, 2.1]#[2.5/5, 7.5/5] #4.2,10.4
    """------------------------------------"""
    
    # Only use the data that contains a speed and an image
    df = df[(df['DateDriven_r'] != "-")].reset_index(drop=True)
    print(df)
    
    # Remover images that could not be obtained
    #roads_unknownimage = pd.DataFrame(df[df['image'] == "-"].groupby(by=['u', 'v', 'key']).agg({'image': 'first'})).reset_index(drop = True)
    #print(roads_unknownimage)
    df = df[(df['image'] != "-")].reset_index(drop = True)
    print(df)
    
    from pandarallel import pandarallel
    # initialize pandarallel
    pandarallel.initialize(progress_bar=True, nb_workers = 4)
    # define a function to count the occurrences of [10, 10, 10]
    def count_sublists(l):
        unnested_l = [elem for sublist in eval(l) for elem in sublist]
        return unnested_l.count([10, 10, 10])
    # apply the function to each row in parallel
    df['Faults'] = df['image'].parallel_apply(count_sublists)
    
    # Remover images that contain no data values (not completely known)
    # For the amount of images that is not fully found and their visualization
    #roads_partially_unknown = pd.DataFrame(df[df.Faults.isin(range(1,100))].groupby(by=['u', 'v', 'key']).agg({'Faults': 'first', 'image':'first'})).reset_index(drop = True)
    #print(roads_partially_unknown)
    #plotImages(test.image[0:40], labels = test.Faults[0:40]) # Default value is [10, 10, 10], which is just black
    df = df[(df['Faults'] == 0)].reset_index(drop = True)
    print(df)

    # We assume that having a speed <=3 corresponds to not driving at all
    df['Speed'] = pd.to_numeric(df['Speed'])
    df = df[(df['Speed']>3)].reset_index(drop = True)
    print(df)
 
    df['DateDriven_string'] = df['DateDriven']
    df['DateDriven'] = pd.to_datetime(df['DateDriven'])
    print(df)
   
    # How to save the df_roads
    save_name = 'rain_'+str(ENV_VARIABLES[0].split("_")[1][:-1])+"h_"+str(len(RAIN_CATS)+1)+ "_"+ '_'.join(str(x) for x in RAIN_CATS) + "_0513.csv"

    df['edge'] = "(" + df['u'].map(str) + ',' + df['v'].map(str) + ',' + df['key'].map(str) + ")"
    df['hour_of_day'] = pd.to_datetime(df['DateDriven']).dt.hour
    df['AmountObs'] = 1
    
    rain_metric = ENV_VARIABLES[0]
    rain_metric_name = rain_metric + "_cat_" + str(len(RAIN_CATS)+1)

    ############ Create the ENV variables if non-existing ############
    #df['rain_5h_cat_3'] = pd.cut(df.rain_5h, bins = 3)
    #df['rain_5h_cat_3'] = pd.qcut(df.rain_5h, q = 3, duplicates='raise')
    def detcat_rain(r, rain_cats):
        i = 0
        while True:
            try: 
                if r < rain_cats[i]: return i
            except: return i
            i+=1 
    df[rain_metric_name] = df[rain_metric].apply(lambda r: detcat_rain(r, rain_cats = RAIN_CATS))
    print(df.groupby(by=[rain_metric_name]).agg({'edge':'count'})) # Check the distribution
    df['dark'] = np.where((df.hour_of_day >= 20) | (df.hour_of_day <= 5), 0, 1)
    ##################################################################
    
    df = df.sort_values(by = ['DateDriven']).reset_index(drop =True)

    # Which variables to use
    VARS_TO_USE = {'edge':'first', 
     'geometry':'first', 
     'highway':'first', 
     'oneway':'first', 
     'new':'first', 
     'image':'first', 
     rain_metric:'first',
     rain_metric_name: 'first',
     'dark': 'first',
     'vegetation_percentage': 'first',
     'swir_min': 'first',
     'swir_max': 'first',
     'swir_mean': 'first', 
     'DateDriven':lambda x:list(x), 
     'DateDriven_string': lambda x:list(x),
     'Speed':lambda x:list(x),
     'AmountObs':'count'}
    

    # Group dataframe by road and circumstance (we want a row per road NOT per speed registration)
    df_roads = pd.DataFrame(df.groupby(by = ['edge']+[rain_metric_name]+[ENV_VARIABLES[1]], sort = False).agg(VARS_TO_USE)).reset_index(drop = True)
    df_roads.to_csv("CustomData/df_roads_pre_"+save_name, sep = ";")
    
    # Add location information of the road
    df_roads['centroid'] = df_roads['geometry'].apply(lambda x: x.centroid)
    df_roads['midlon'] = df_roads['centroid'].apply(lambda p: p.x)
    df_roads['midlat'] = df_roads['centroid'].apply(lambda p: p.y)

    # Add information about the speed
    df_roads['Avg_Speed'] = df_roads['Speed'].apply(lambda x: sum(x)/len(x))
    
    # Adjust the average speed based on trajectories (in one go)
    df_roads['Avg_Speed_adjusted'] = None
    for i in range(0,len(df_roads)):
        dates = df_roads.DateDriven[i]
        speeds = df_roads.Speed[i]
        cuts = list(np.where([True] + [abs((dates[i+1] - dates[i]).total_seconds())>120 for i in range(0,len(dates)-1)])[0]) + [len(dates)]
        S = []
        for c in range(0, len(cuts)-1):
            cut_speed = np.mean(speeds[cuts[c]:cuts[c+1]])
            S += [cut_speed]
        df_roads.loc[i, 'Avg_Speed_adjusted'] = np.round(np.mean(S),0)
    df_roads['Avg_Speed_adjusted'] = df_roads.Avg_Speed_adjusted.map(int)
    
    #########################################################################################################################
    ############################################# HIGHWAY AND SPEED CATS (NOT USED) #########################################
    #########################################################################################################################
    # For information about the highway type, we only consider single labeled road types and create a continuous variable out of these
    hw_types = ['trunk', 'primary', 'secondary', 'tertiary','unclassified', 'residential', 'service', 'track', 'path']
    hw_groups = [['trunk', 'primary', 'secondary'], ['tertiary','unclassified', 'residential'], ['service', 'track', 'path']]
    def detcat_hw(hw):
        global hw_types
        if hw is None: return None
        elif hw in hw_types[0]: return 0
        elif hw in hw_types[1]: return 1
        elif hw in hw_types[2]: return 2
        elif hw in hw_types[3]: return 3
        elif hw in hw_types[4]: return 4
        elif hw in hw_types[5]: return 5
        elif hw in hw_types[6]: return 6
        elif hw in hw_types[7]: return 7
        elif hw in hw_types[8]: return 8
        else: return None
    def detcat(hw):
        global hw_groups
        if hw is None: return None
        elif hw in hw_groups[0]: return 0
        elif hw in hw_groups[1]: return 1
        elif hw in hw_groups[2]: return 2
        else: return None
    df_roads['highway_cat'] = df_roads.highway.apply(lambda p: detcat(p)).map(str)
    df_roads['highway_cat_9'] = df_roads.highway.apply(lambda p: detcat_hw(p)).map(str)

    # Add a speed category label
    df_roads['Speed_cat'] = pd.qcut(np.array(df_roads['Avg_Speed_adjusted']), NUM_SPEED_CATS)
    df_roads['Speed_cat_label'] = pd.qcut(np.array(df_roads['Avg_Speed_adjusted']), NUM_SPEED_CATS, labels = range(0, NUM_SPEED_CATS))
    #########################################################################################################################
    
    # Only save relevant features
    df_roads = df_roads[['edge','highway_cat', 'highway_cat_9','Speed_cat_label', 'image', 'Avg_Speed_adjusted', 'AmountObs', 'midlon', 'midlat', rain_metric, rain_metric_name, 'dark', 'vegetation_percentage', 'swir_min', 'swir_max', 'swir_mean']]
    print(df_roads)
    # Save the custom-made dataframe
    if not os.path.exists("CustomData"): os.makedirs("CustomData")
    df_roads.to_csv("CustomData/"+save_name, sep = ";", index = False)
############################################################################################
############################################################################################

############################################################################################
################################### CREATE WEATHER DF ######################################
############################################################################################
"""
 Here, we create the data that is focused on the weather features. 

 Input : df_prepared_3
 Output : df_roads_weather_impact
"""
if CREATE_WEATHER_DF:
    """------------- Settings -------------"""
    RAIN_CATS = [2.5, 7.5]
    # TODO.Currently, this is only done for 5h (not used in the actual analysis)
    """------------------------------------"""
    
    # Only use the data that contains a speed and an image
    df = df[(df['DateDriven_r'] != "-") & (df['image'] != "-")].reset_index(drop=True)
    
    # Add two columns
    df['edge'] = "(" + df['u'].map(str) + ',' + df['v'].map(str) + ',' + df['key'].map(str) + ")"
    df['hour_of_day'] = pd.to_datetime(df['DateDriven_r']).dt.hour
    
    # Extract only the relevant columns that we are going to use
    VARS_TO_USE = ['edge', 'geometry', 'highway', 'Speed', 'hour_of_day', 'wind_speed', 'wind_gust', 'weather_description', 'rain_1h', 'rain_3h', 'rain_5h','rain_10h', 'rain_24h', 'image']
    df = df.loc[:,VARS_TO_USE]
    
    def detcat_rain_5h(r, rain_cats):
        i = 0
        while True:
            try: 
                if r < rain_cats[i]: return i
            except: return i
            i+=1
    df['rain_5h_cat_3'] = df.rain_5h.apply(lambda r: detcat_rain_5h(r, rain_cats = RAIN_CATS))
    df['dark'] = np.where((df.hour_of_day >= 20) | (df.hour_of_day <= 5), 'Dark', 'Light')
    
    # Add location information of the road
    df['centroid'] = df['geometry'].apply(lambda x: x.centroid)
    df['midlon'] = df['centroid'].apply(lambda p: p.x)
    df['midlat'] = df['centroid'].apply(lambda p: p.y)
    
    # Only consider single labeled road types and create a continuous variable out of these
    hw_types = ['trunk', 'primary', 'secondary', 'tertiary','unclassified', 'residential', 'service', 'track', 'path']
    hw_groups = [['trunk', 'primary', 'secondary'], ['tertiary','unclassified', 'residential'], ['service', 'track', 'path']]
    def detcat_hw(hw):
        global hw_types
        try: 
            if hw in hw_types[0]: return 0
            elif hw in hw_types[1]: return 1
            elif hw in hw_types[2]: return 2
            elif hw in hw_types[3]: return 3
            elif hw in hw_types[4]: return 4
            elif hw in hw_types[5]: return 5
            elif hw in hw_types[6]: return 6
            elif hw in hw_types[7]: return 7
            elif hw in hw_types[8]: return 8
            else: return None
        except: return None
    def detcat(hw):
        global hw_groups
        try: 
            if hw in hw_groups[0]: return 0
            elif hw in hw_groups[1]: return 1
            elif hw in hw_groups[2]: return 2
            else: return None
        except: return None
    df['highway_cat'] = df.highway.apply(lambda p: detcat(p)).map(str)
    df['highway_cat_9'] = df.highway.apply(lambda p: detcat_hw(p))
    
    # By rounding to the hour, we may get some duplicate rows
    df.drop_duplicates(subset = ['edge', 'highway', 'Speed', 'hour_of_day', 'wind_speed', 'wind_gust', 'weather_description', 'rain_1h', 'rain_3h', 'rain_5h','rain_10h', 'rain_24h'], keep='first', inplace = True)
    df.reset_index(drop = True, inplace = True)
    
    # Save the data 
    if not os.path.exists("CustomData"): os.makedirs("CustomData")
    df.to_csv("CustomData/df_roads_weather_impact.csv", sep = ";")
############################################################################################
############################################################################################

############################################################################################
##################################### CREATE RAIN DF #######################################
############################################################################################
"""
 Here, we examine, for each road type, what the average speed (NOT ADJUSTED) is during 
 specific circumstances (e.g., rain, darkness). 

 Input : df_prepared_3
 Output : df_roads_rain_impact
"""
if CREATE_RAIN_DF:
    """------------- Settings -------------"""
    # TODO.Currently, this is only done for 5h (not used in the actual analysis)
    """------------------------------------"""
    RAIN_CATS = [0.8, 2.1]
    rain_metric = 'rain_1h'
    
    # Only use the data that contains a speed and an image
    df = df[(df['DateDriven_r'] != "-") & (df['image'] != "-")].reset_index(drop=True)
    
    # Add two columns
    df['edge'] = "(" + df['u'].map(str) + ',' + df['v'].map(str) + ',' + df['key'].map(str) + ")"
    df['hour_of_day'] = pd.to_datetime(df['DateDriven_r']).dt.hour
    
    # Extract only the relevant columns that we are going to use
    VARS_TO_USE = ['edge', 'highway', 'Speed', 'hour_of_day', 'wind_speed', 'wind_gust', 'weather_description', 'rain_1h', 'rain_3h', 'rain_5h','rain_10h', 'rain_24h']
    df = df.loc[:,VARS_TO_USE]
    
    df['dark'] = np.where((df.hour_of_day >= 20) | (df.hour_of_day <= 5), 'Dark', 'Light')
    def detcat_rain(r, rain_cats):
        i = 0
        while True:
            try: 
                if r < rain_cats[i]: return i
            except: return i
            i+=1 
    df['Rain_cat'] = df[rain_metric].apply(lambda r: detcat_rain(r, rain_cats = RAIN_CATS))
    #df['Rain_cat'] = pd.cut(np.array(df['rain_1h']), AMOUNT_RAIN_CAT)

    # Only consider single labeled road types and create a continuous variable out of these
    hw_types = ['trunk', 'primary', 'secondary', 'tertiary','unclassified', 'residential', 'service', 'track', 'path']
    hw_groups = [['trunk', 'primary', 'secondary'], ['tertiary','unclassified', 'residential'], ['service', 'track', 'path']]
    def detcat_hw(hw):
        global hw_types
        try: 
            if hw in hw_types[0]: return 0
            elif hw in hw_types[1]: return 1
            elif hw in hw_types[2]: return 2
            elif hw in hw_types[3]: return 3
            elif hw in hw_types[4]: return 4
            elif hw in hw_types[5]: return 5
            elif hw in hw_types[6]: return 6
            elif hw in hw_types[7]: return 7
            elif hw in hw_types[8]: return 8
            else: return None
        except: return None
    def detcat(hw):
        global hw_groups
        try: 
            if hw in hw_groups[0]: return 0
            elif hw in hw_groups[1]: return 1
            elif hw in hw_groups[2]: return 2
            else: return None
        except: return None
    df['highway_cat'] = df.highway.apply(lambda p: detcat(p)).map(str)
    df['highway_cat_9'] = df.highway.apply(lambda p: detcat_hw(p))
    # df = df[df.highway.isin(hw_types)].reset_index(drop = True)
    
    #print(df.Speed)
    #print(df['Speed'].isna().sum())
    df['Speed'] = pd.to_numeric(df['Speed'])
    

    # Group by such that comparisons can be made
    df_relevant = df.groupby(by=['Rain_cat', 'highway_cat_9']).agg(Speed=('Speed', 'mean'),highway_cat_9=('highway_cat_9', 'first'),count=('Speed', 'count'))#, 'rain_1h': 'mean', 'rain_3h': 'mean', 'rain_5h': 'mean', 'rain_10h': 'mean', 'rain_24h': 'mean'})
    print(df_relevant.Speed)
    
    stop
    # Sort dataframe to get an even better representation
    df_relevant.reset_index(inplace = True)
    df_relevant.sort_values(by = ['highway_cat_9', 'Rain_cat', 'dark'], ascending = True, inplace = True)
    df_relevant.reset_index(inplace = True, drop = True)
    df_relevant['Rain_cat'] = df_relevant.Rain_cat.map(str)

    # Save custom Df
    if not os.path.exists("CustomData"): os.makedirs("CustomData")
    df_relevant.to_csv("CustomData/df_roads_rain_impact.csv", sep = ";")
############################################################################################
############################################################################################


############################################################################################
#################################### CREATE FULL DF ########################################
############################################################################################
"""
 Input : df_prepared_3
 Output : df_roads
"""
if CREATE_FINAL_COMPARE:
    df['edge'] = "(" + df['u'].map(str) + ',' + df['v'].map(str) + ',' + df['key'].map(str) + ")"
    df = df[df['image'] != "-"].reset_index(drop=True)
    
    # Which variables to use
    VARS_TO_USE = {'edge': 'first',
     'u': 'first', 
     'v': 'first', 
     'key': 'first', 
     'geometry': 'first',
     'image': 'first',
     'vegetation_percentage': 'first', 
     'swir_min': 'first',
     'swir_max': 'first',
     'swir_mean': 'first'}
    
    # Add edge info
    df = pd.DataFrame(df.groupby(by = ['edge'], sort = False).agg(VARS_TO_USE)).reset_index(drop = True)
      
    # Add location information of the road
    df['centroid'] = df['geometry'].apply(lambda x: x.centroid)
    df['midlon'] = df['centroid'].apply(lambda p: p.x)
    df['midlat'] = df['centroid'].apply(lambda p: p.y)
    
    VARS_TO_USE = ['edge', 'u', 'v', 'key', 'midlon', 'midlat', 'image', 'vegetation_percentage', 'swir_min', 'swir_max', 'swir_mean']
    df = df.loc[:,VARS_TO_USE]
    df.to_csv('CustomData/df_roads_FULL.csv', sep = ";")
############################################################################################
############################################################################################
""" --------------------------------------------------------------------------------------- """






""" --------------------------------------------------------------------------------------- """
""" -------------------------------- VISUALIZE INFORMATION -------------------------------- """
""" --------------------------------------------------------------------------------------- """
""" -------- Settings ------- """ 
PLOT_SATELLITE_IMAGES = True
if PLOT_SATELLITE_IMAGES:
    ROAD_TYPE = None #None
    AMOUNT = 40
VISUALIZE_SPEEDS = False # Requires df_roads (see above)
if VISUALIZE_SPEEDS:
    ROAD = [0] # 0: road with most speed registrations
    DAYS = [0] # Plot these days (0: day with highest activity)
""" ------------------------- """

############################################################################################
############################### VISUALIZE SATELLITE IMAGES #################################
############################################################################################
"""
 Here, we can visualize some of the satellite images present in the dataframe. We can 
 specify whether we would like to see only roads of a specific type. 

 Input : df_prepared_3
""" 
def plotImages(images, labels = None):
    """
     This procedure can be used to visualize a specific satellite image,
     using the df_satellite dataset (that contains the images themselves)
     a maximum of 40 images can be plotted at the same time
    """
    # Plot max 40 images
    number_figures = len(images)
    
    # Determine widht,height
    max_d = 5
    w = int(min(number_figures, max_d))
    h = int(1 + np.floor((number_figures-1) / max_d))
    try: a = images.index[0]
    except: a = 0
    plt.figure()
    for i in range(0,min(len(images), 40)):
        plt.subplot(h,w,i+1)
        plt.xticks([])
        plt.yticks([])
        if labels is not None:
            plt.title(str(labels[i]))#df_satellite.highway[i])
        plt.grid(False)
        maxValue = np.amax(images[i+a])
        minValue = np.amin(images[i+a])
        plt.imshow(images[i+a])
    plt.show()
if PLOT_SATELLITE_IMAGES:
    # Define a custom converter function that extracts the image from the list
    def image_converter(image_list_str):
        if image_list_str == "-": return image_list_str
        else:
            image_list = eval(image_list_str)
            return [np.array(image).astype(int) for image in image_list]
    def geom_converter(wkt_str):
        return wkt.loads(wkt_str)
    df = pd.read_table("Data/df_prepared_3.csv", sep = ";", index_col = 0, converters={'geometry':geom_converter, 'image': image_converter}, low_memory=False)#, 'image': image_converter
    
    # Group the data by geometry
    df['geometry_wkb'] = df['geometry'].apply(lambda geom: geom.wkb) # Also add the wkb info (just for redundancy)
    df_relevant = pd.DataFrame(df.groupby(by = ['geometry_wkb'], sort = False).agg({'geometry':'first', 'highway':'first', 'image':'first'})).reset_index(drop = True)

    if ROAD_TYPE is not None: df_relevant = df_relevant.loc[df_relevant.highway == ROAD_TYPE].reset_index(drop=True)
    plot_images = range(0,AMOUNT)
    plotImages(list(df_relevant.image[plot_images]) , labels = list(df_relevant.highway[plot_images]))
############################################################################################
############################################################################################

############################################################################################
#################################### VISUALIZE SPEEDS ######################################
############################################################################################
"""
 For a given road, and a given day, we can visualize all the speed registrations of the 
 vehicles. This will create a set of trips that can be visualized over time. 

 Input : rain_1h_3_0.8_2.1 (or similar)
"""
if VISUALIZE_SPEEDS:
    # Load relevant dataframe
    df = pd.read_csv("CustomData/df_roads_pre_rain_1h_3_0.8_2.1.csv", sep = ";", index_col = 0)
    #df['DateDriven'] = pd.to_datetime(df['DateDriven'])
    df['Speed'] = df['Speed']
    #with open(r"CustomData/rain_1h_3_0.8_2.1.pickle", "rb") as input_file: df = pickle.load(input_file)
    df['len_speed'] = df["Speed"].apply(lambda ls: len(eval(ls)))
    df.sort_values(by = ['len_speed'], ascending = False, inplace = True) # row 0 is road with most speed registrations)
    df.reset_index(drop = True, inplace = True)

    for r in ROAD:
        print("Dealing with a:", df.highway[r])
        # Create a df for each individual road (with the datedriven and the speed grouped per day) (and some other relevant columns)
        df_temp = pd.DataFrame()
        df_temp['DateDriven'] = pd.to_datetime(eval(df.DateDriven_string[r]))
        df_temp['Date'] = df_temp.DateDriven.apply(lambda d: d.date())
        df_temp['Speed'] = eval(df.Speed[r])
        df_temp = df_temp.sort_values(by = ['DateDriven'], inplace = False).reset_index(drop = True)
        
        # Group by date
        df_temp = pd.DataFrame(df_temp.groupby(by = ['Date']).agg({'Date':'first', 'DateDriven': lambda x: list(x), 'Speed': lambda x: list(x)})).reset_index(drop = True)
        df_temp['len_speed'] = df_temp["Speed"].apply(lambda ls: len(ls))
        
        #OPTIONAL sort according to the amount of something
        df_temp.sort_values(by = ['len_speed'], ascending = False, inplace= True)
        df_temp.reset_index(drop=True, inplace=True)
        
        # plot the figure (for each road) sequentially
        plt.figure(figsize = (13.2,4))
        for d in DAYS:
            dates = df_temp.DateDriven[d]
            speeds = df_temp.Speed[d]
            cuts = list(np.where([True] + [abs((dates[i+1] - dates[i]).total_seconds())>120 for i in range(0,len(dates)-1)])[0]) + [len(dates)]
            # Add the plots
            for i in range(0, len(cuts)-1):
                plt.plot(dates[cuts[i]:cuts[i+1]], speeds[cuts[i]:cuts[i+1]], '-o')
        plt.tight_layout()
        plt.show()
############################################################################################
############################################################################################
""" --------------------------------------------------------------------------------------- """