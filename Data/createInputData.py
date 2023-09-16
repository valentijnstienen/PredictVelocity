import pandas as pd
import numpy as np
import geopandas as gp
import pickle
import ee
import folium
import osmnx as ox
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from datetime import date, time, datetime, timedelta
from shapely.geometry import Polygon, LineString, Point
from math import radians, cos, sin, asin, sqrt, atan2, pi
from scipy import ndimage
from statistics import mean
from satFunctions import get_sentinel_image

from shapely import wkt

with open('mapbox_accesstoken.txt') as f: mapbox_accesstoken = f.readlines()[0]

"""
 In this file, input data is created. There are multiple input data sets that can be created. 
"""

""" ------------ Settings ------------ """
PREPARE_EDGE_DF = True
COMBINE_EDGEWEATHER = False
if COMBINE_EDGEWEATHER: 
    WEATHER_SOURCE = "OpenWeather" # or "Wunderground" (need to be available!)
COMBINE_EDGESATELLITE = False
""" ---------------------------------- """

############################################################################################
############################# 1. PREPARE THE EDGE DATASET ##################################
############################################################################################
"""
 First, we create a dataframe of all the edges that came from the network extension model. 
 We remove the unused edges (later we may want to include these) and we create a row 
 (observation) for each speed registration received. 
 
 Input : Edges_0_14960
 Output : df_prepared_1
"""
if PREPARE_EDGE_DF:
    print("------- PREPARING EDGE DATASET -------")
    print()
    def haversine(lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified as lat/long combinations
        """
        # convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles
        return c * r * 1000 
    def computeLengthLinestring(line, method = "haversine"): 
        """
         Returns the length of the linestring [line], using the [method].
    
         Parameters
         ----------
         line : LineString
         method : string 
            determines how to compute the length between points (e.g, haversine or euclidean)

         Returns
         -------
         distance : float
             length of the [line]
    
        """
        # Extract all coordinates
        numCoords = len(line.coords) - 1
    
        distance = 0
        for i in range(0, numCoords):
            point1 = line.coords[i]
            point2 = line.coords[i + 1]
            if method == "haversine": distance += haversine(point1[0], point1[1], point2[0], point2[1])
            else: distance += sqrt((point1[0]-point2[0])**2 + (point1[1] - point2[1])**2)
        return distance
    
    # Load the data (edges)
    df = pd.read_table('Data/_Graphs_SIMPLE/Edges.csv', sep=";", index_col=0, low_memory=False)
    df['geometry'] = gp.GeoSeries.from_wkt(df['geometry'])
    
    df = df[df.length > 0] # Columns of one point are useless. Note that these points are formed while splitting an existing edge into two edges. In this case, the information of this specific point will also be available in the other part of the edge (which will typically not be a point (otherwise, still useless.))
    
    #dat_vels = df['DatesVelocities'].apply(lambda x: [y for y in x.split("|") if y] if x else x)
    dat_vels = df['DatesVelocities'].apply(lambda x: x.split("|") if isinstance(x, str) else [])
    dats = [item for sublist in list(dat_vels.apply(lambda x: x[1::2] if x else "-")) for item in sublist]
    vels = [item for sublist in list(dat_vels.apply(lambda x: x[2::2] if x else "-")) for item in sublist] 
    len_datvels = df['DatesVelocities'].apply(lambda x: len([y for y in x.split("|") if y])/2 if isinstance(x, str) else 1)
    
    df_new = df.loc[df.index.repeat(len_datvels), :]
    df_new["DateDriven"] = dats
    df_new['DateDriven_r'] = df_new['DateDriven'].apply(lambda x: pd.to_datetime(x).round('1h') if x != '-' else x)
    
    # Target (Speed variable)
    df_new["Speed"] = vels
    df_new.Speed = df_new['Speed'].apply(lambda x: int(float(x)) if x != '-' else x)  
    
    # Define the features that are used
    features_used = ['u', 'v', 'key', 'osmid', 'highway', 'maxspeed', 'oneway', 'new', 'geometry', 'length', 'DateDriven', 'DateDriven_r', 'Speed'] # DateDriven
    df_new = df_new.astype({'length': int})
    
    # To obtain all figures (also those not driven)
    df_new = df_new[features_used]
    df_new.reset_index(drop=True, inplace = True)
    
    # Save df
    df_new.to_csv("Data/df_prepared_1.csv", sep = ";")
    print("------- EDGE DATASET PREPARED --------")
############################################################################################
############################################################################################

# NOW CREATE THE WEATHER AND SATELLITE DATASET! 

############################################################################################
############################ 3. COMBINE WEATHER AND EDGE DATA ##############################
############################################################################################
"""
 Next, we add all weather features to the edge data. So each observation of a speed of 
 a truck is now combined with the weather at that point in time (and maybe before). 

 Input : df_prepared_1, df_weather_OD OR df_weather_WU (depending on selection)
 Output : df_prepared_2
"""
if COMBINE_EDGEWEATHER:
    print("------- COMBINE EDGE AND WEATHER DATASET --------")
    print()
    
    # Load data
    df = pd.read_table("Data/df_prepared_1.csv", sep = ";", index_col = 0, low_memory = False)
    df.sort_values(by = 'DateDriven_r', inplace = True)
    df.reset_index(drop=True, inplace = True)
    amount_repetitions = list(df.groupby(by = ['DateDriven_r'], sort = False).agg({'key': 'count'}).key)
    print("Amount of weathers to be processed: ", len(amount_repetitions))
    
    # Load pre-run weather data (that contains all datetimes in df)
    if WEATHER_SOURCE == "OpenWeather": df_weather = pd.read_table('Data/df_weather_OD.csv', index_col = 0, sep = ";")
    elif WEATHER_SOURCE == "Wunderground": df_weather = pd.read_table('Data/df_weather_WU.csv', sep = ";", index_col = 0)
    
    i = 0
    for w in range(0,len(amount_repetitions)):
        # Print progress
        if w%500 == 0: print("Weather:", w)
        # How many rows have the same whether
        reps = amount_repetitions[w]
        
        # Determine the time 
        weather_time = df.loc[i, 'DateDriven_r']
        if weather_time == "-": 
            i+= reps
            continue
            
        # Add the weather information
        weather_row = df_weather.loc[df_weather.dt_iso_AsiaJakarta == weather_time].reset_index(drop=True).loc[0:1,df_weather.columns[3::]]
        df.loc[i:(i+reps-1), weather_row.columns] = weather_row.values
        
        i += reps
    
    # Save the data
    df.to_csv('Data/df_prepared_2.csv', sep = ";")
    print("------- EDGE AND WEATHER DATASET COMBINED -------")       
############################################################################################
############################################################################################

############################################################################################
########################## 6. COMBINE SATELLITE AND EDGE DATA ##############################
############################################################################################
"""
 Next, we add all satellite features to the edge data. So each observation of a speed of 
 a truck is now combined with the satellite info of that point. 

 Input : df_prepared_1 (if weather NOT included), df_prepared_2 (if weather included), df_satellite
 Output : df_prepared_3
"""
if COMBINE_EDGESATELLITE:
    print("------- COMBINE EDGE AND SATELLITE DATASET --------")
    print()
    def computeLengthLinestring(line, method = "haversine"): 
        """
         Returns the length of the linestring [line], using the [method].
    
         Parameters
         ----------
         line : LineString
         method : string 
            determines how to compute the length between points (e.g, haversine or euclidean)

         Returns
         -------
         distance : float
             length of the [line]
    
        """
        # Extract all coordinates
        numCoords = len(line.coords) - 1
    
        distance = 0
        for i in range(0, numCoords):
            point1 = line.coords[i]
            point2 = line.coords[i + 1]
            if method == "haversine": distance += haversine(point1[0], point1[1], point2[0], point2[1])
            else: distance += sqrt((point1[0]-point2[0])**2 + (point1[1] - point2[1])**2)
        return distance
    
    def find_line_id(df, line):
        ids = 0
        while True:
            check_geom = df.geometry[ids] 
            if line.equals_exact(check_geom, tolerance = 0.1): return ids
            ids += 1

    # Load data
    def geom_converter(wkt_str):
        return wkt.loads(wkt_str) 
    df = pd.read_table('Data/df_prepared_2.csv', sep = ";", index_col = 0, converters={'geometry':geom_converter}, low_memory = False)
    df['geometry_wkb'] = df['geometry'].apply(lambda geom: geom.wkb)
    df.sort_values(by = 'geometry_wkb', inplace = True)
    df.reset_index(drop=True, inplace = True)
    amount_repetitions = list(df.groupby(by = ['geometry_wkb'], sort = False).agg({'key': 'count'}).key)
    print("Amount of roads to be processed: ", len(amount_repetitions))
    df.drop(columns=['geometry_wkb'], inplace = True)
    
    # Load pre-run satellite images
    def image_converter(image_list_str):
        if image_list_str == "-": return image_list_str
        else:
            image_list = eval(image_list_str)
            return [np.array(image) for image in image_list]
    df_satellite = pd.read_table("Data/df_satellite.csv", sep = ";", index_col = 0, converters={'geometry':geom_converter})
    
    for c in ['image', 'cloud_cover', 'vegetation_percentage', 'swir_min', 'swir_max', 'swir_mean', 'nimages', 'generation_time']:
        df[c] = None
    
    i = 0
    for r in range(0,len(amount_repetitions)):
        # Print progress
        if r%100 == 0: print("Road:", r)
        # How many rows have the same satellite image
        reps = amount_repetitions[r]
        
        # Determine the geometry
        geom = df.loc[i, 'geometry']
        
        try: 
            ids = find_line_id(df = df_satellite, line = geom)
            df.loc[i:(i+reps-1), ['image', 'cloud_cover', 'vegetation_percentage', 'swir_min', 'swir_max', 'swir_mean', 'nimages', 'generation_time']] = [df_satellite.loc[ids, ['image', 'cloud_cover', 'vegetation_percentage', 'swir_min', 'swir_max', 'swir_mean', 'nimages', 'generation_time']]]*reps
        except: 
            print("Image",r,": no image found (yet)...")
            print(reps)
            df.loc[i:(i+reps-1), ['image', 'cloud_cover', 'vegetation_percentage', 'swir_min', 'swir_max', 'swir_mean', 'nimages', 'generation_time']] = [[None, None, None, None, None, None, None, None]]*reps
        
        i += reps
        
        # Save the data while running
        if r%100 == 0: df.to_csv('Data/df_prepared_3.csv', sep = ";")
            
    # Save final df
    df.to_csv('Data/df_prepared_3.csv', sep = ";")
    print("------- EDGE AND SATELLITE DATASET COMBINED -------")
############################################################################################
############################################################################################