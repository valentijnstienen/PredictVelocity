import pandas as pd
import numpy as np
import geopandas as gp

import osmnx as ox
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from scipy import ndimage
from statistics import mean

from shapely import wkt

from math import radians, cos, sin, asin, sqrt, atan2, pi
from shapely.geometry import Polygon, LineString, Point
from satFunctions import get_sentinel_image

with open('mapbox_accesstoken.txt') as f: mapbox_accesstoken = f.readlines()[0]
    
# If the function has been interrupted and we have multiple df_satellite files, we can combine them here
# SatelliteDB_1 = pd.read_table("Data/df_satellite_PART1.csv", sep = ";", index_col = 0)
# SatelliteDB_2 = pd.read_table("Data/df_satellite_PART2.csv", sep = ";", index_col = 0)
# SatelliteDB_1.iloc[38400::,] = SatelliteDB_2.iloc[38400::,]
# SatelliteDB_1.to_csv('Data/df_satellite.csv', sep=";")
# stop

# Load the satellite df
# Define a custom converter function that extracts the image from the list
# def image_converter(image_list_str):
#     if image_list_str == "-": return image_list_str
#     else:
#         image_list = eval(image_list_str)
#         return [np.array(image).astype(int) for image in image_list]
# def geom_converter(wkt_str):
#     return wkt.loads(wkt_str)
# SatelliteDB = pd.read_table("Data/df_satellite.csv", sep = ";", index_col = 0, converters={'geometry':geom_converter, 'image': image_converter})
# print(SatelliteDB)


##################################################
################ FUNCTIONS USED ##################
##################################################
def computeBearing(start_point, end_point):
    """
    This function computes the bearing between two points: [start_point] and [end_point].

    Parameters
    ----------
    start_point : Point(x,y)
    end_point : Point(x,y)

    Returns
    -------
    brng_new : float

    """
    brng_new = (180/pi) * atan2(end_point.x-start_point.x, end_point.y - start_point.y)
    if brng_new >= 0: return np.round(brng_new,0)
    else: return np.round(brng_new+360,0)

def headings_linestring(line):
    # Extract all coordinates
    numCoords = len(line.coords) - 1
    
    bearings = []
    for i in range(0, numCoords):
        point1 = Point(line.coords[i])
        point2 = Point(line.coords[i + 1])
        bearings += [computeBearing(point1,point2)]
    
    return bearings
    
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
        print(i)
        plt.subplot(h,w,i+1)
        plt.xticks([])
        plt.yticks([])
        if labels is not None:
            plt.title(str(labels[i]))#df_satellite.highway[i])
        plt.grid(False)
        maxValue = np.amax(images[i+a])
        minValue = np.amin(images[i+a])
        
        print(maxValue)
        print(minValue)
        
        print(images[i+a])
        plt.imshow(images[i+a])
    plt.show()

# print(SatelliteDB.iloc[1002:1003,:])
# plotImages(SatelliteDB.image[1002:1003])
# stop
  
def get_segment_distances(linestring):
    points = [y for y in line.coords] 
    distances = []
    for p in range(0,len(points[:-1])):
        distances += [Point(points[p]).distance(Point(points[p+1]))]

    return np.cumsum(distances)

def get_representative_point_road(road_geom, bearings, amount_repr_points, min_dist):
    """
    
    
    """
    # Determine at what distances there are representative points. They should be at least 100m apart, otherwise we may have overlapping pictures. 
    edge_length = road_geom.length
    distances = np.linspace(0, edge_length, amount_repr_points + 2) # Include end points (no representative points)
    if edge_length < 100: distances = [0, 0.5*edge_length,edge_length]
    elif distances[1]-distances[0] < min_dist: distances = list(np.arange(0, road_geom.length, min_dist))+[road_geom.length]
    distances = distances[1:-1]
    
    # Determine on which line segments the representative points lie
    a = get_segment_distances(road_geom)
    indices_repr_points = [np.where(a>d)[0][0] for d in distances]
    
    # Find the representative points
    repr_points = [road_geom.interpolate(distance) for distance in distances]
    repr_bearings = [bearings[i] for i in indices_repr_points]
    
    # Start picturing the road
    print("Examine", len(repr_bearings), "pictures.")
    images, removed_indices = [], []
    cloud_cover, generation_time, vegetation_percentage, swir_min, swir_max, swir_mean = [], [], [], [], [], []
    
    count = 0
    for p in repr_points:
        p_ll = ox.projection.project_geometry(p, crs = used_crs, to_latlong=True)[0]
        print(p_ll)
        RADIUS = 80
        image, cc, gt, vp, smin, smax, smean = get_sentinel_image(p_ll, radius = RADIUS, visualize =False, vis_name = "test")
        
        
        if image is None: removed_indices += [count]
        else: 
            image = image[:int(RADIUS*2/10), :int(RADIUS*2/10), :] #Remove potential edges
            images += [image]
            cloud_cover += [cc]
            generation_time += [gt]
            vegetation_percentage += [vp]
            swir_min += [smin]
            swir_max += [smax]
            swir_mean += [smean]
        count+=1
    
    if len(images) < 1:
        print("Unfortunately, no (mean) image could have been found for this road...")
        return None, None, None, None, None, None, None, None, None, None
    
    mean_cloud_cover, mean_vegetation_percentage, mean_swir_min, mean_swir_max, mean_swir_mean = np.mean(cloud_cover), np.mean(vegetation_percentage), np.mean(swir_min), np.mean(swir_max), np.mean(swir_mean) 
    
    repr_bearings = [ele for ele in repr_bearings if ele not in removed_indices]
    repr_points = [ele for ele in repr_points if ele not in removed_indices]

    # Rotate the image to align the headings
    rotated_images = []
    for image, brng in zip(images,repr_bearings):
        rotated_images += [ndimage.rotate(image, brng, reshape=False)]
    
    # Plot the rotated image(s)
    # plotImages(rotated_images)
    
    # Composite the mean image 
    mean_image = np.round(np.mean(rotated_images, axis=0),0).astype(int)
    mean_image = mean_image[3:13,3:13,:]

    # Plot if wanted
    #im = images + rotated_images + [mean_image]
    #plotImages(im)

    return repr_points, repr_bearings, mean_image, int(mean_cloud_cover), int(mean_vegetation_percentage), int(mean_swir_min), int(mean_swir_max), int(mean_swir_mean), str(generation_time), len(generation_time)

def postP(x):
    if x is not None: 
        if len(x) == 1: return x[0]
    return x
##################################################

# Load the data (this data contains all the roads)
df = pd.read_table('Data/_Graphs_SIMPLE/Edges.csv', sep=";", index_col=0, low_memory=False)
df['geometry'] = df['geometry'].apply(lambda geom: wkt.loads(geom))
df['geometry_wkb'] = df['geometry'].apply(lambda geom: geom.wkb) # Also add the wkb info (just for redundancy)

# Adjust dataframe
used_crs = "+proj=utm +zone=48 +ellps=WGS84 +datum=WGS84 +units=m +no_defs +type=crs"
df = gp.GeoDataFrame(df, crs=used_crs, geometry=df.geometry)
# Determine the angles of line pieces in a geometry. Needed for rotating the images of the road.
df["bearings"] = df["geometry"].apply(lambda ls: headings_linestring(ls))

# Set up the SatelliteDB
SatelliteDB = pd.DataFrame(columns = ['geometry', 'geometry_wkb', 'image', 'cloud_cover', 'vegetation_percentage', 'swir_min', 'swir_max', 'swir_mean', 'nimages'])
SatelliteDB.geometry = df.geometry
SatelliteDB.geometry_wkb = df.geometry_wkb

TEST_CASE = True
AMOUNT_REPRESENTATIVE_POINTS = 5
if TEST_CASE: 
    TEST_ROADS = [14]
    for tr in TEST_ROADS:
        line = df.loc[tr, 'geometry']
        repr_points, repr_bearings = get_representative_point_road(road_geom = line, bearings = df.bearings[tr], amount_repr_points = AMOUNT_REPRESENTATIVE_POINTS, min_dist = 100)[0:2]

        # Plot the road
        fig = go.Figure()
        # Plot the edge
        line = ox.projection.project_geometry(line, crs = used_crs, to_latlong=True)[0]
        edge_points_list = [y for y in line.coords]
        edge_latitudes, edge_longitudes, edge_notes, ind = [], [], [], 0
        count = 0
        for e in edge_points_list:
            lons, lats = zip(e)
            edge_longitudes = edge_longitudes + list(lons)
            edge_latitudes = edge_latitudes + list(lats)
            edge_notes = edge_notes + len(lats)*[ind]
            ind = ind + 1
            if count == 0:
                fig.update_layout(mapbox1 = dict(center = dict(lat=mean(edge_latitudes), lon=mean(edge_longitudes)), accesstoken = mapbox_accesstoken, zoom = 14),margin = dict(t=10, b=0, l=10, r=10),showlegend=False,mapbox_style="satellite") # or use satellite as mapbox_style
            fig.add_trace(go.Scattermapbox(mode='lines', lat=edge_latitudes, lon=edge_longitudes, text = edge_notes, visible = True, marker = {'size' : 30, 'color': 'yellow', 'allowoverlap': True}))
            count+=1
        # Plot the representative points
        for p in range(0,len(repr_points)):
            p_ll = ox.projection.project_geometry(repr_points[p], crs = used_crs, to_latlong=True)[0]
            fig.add_trace(go.Scattermapbox(mode='markers', lat=[p_ll.y], lon=[p_ll.x], text = "Bearing: " + str(repr_bearings[p]), visible = True, marker = {'size' : 30, 'color': 'orange', 'allowoverlap': True}))
        fig.show()

for r in range(0, len(df)):
    print('Road', r)
    line = df.loc[r, 'geometry']
    bearings = df.loc[r, 'bearings']
    _, _, image, cloud_cover, vegetation_percentage, swir_min, swir_max, swir_mean, generation_time, nimages = get_representative_point_road(road_geom=line, bearings=bearings, amount_repr_points=AMOUNT_REPRESENTATIVE_POINTS, min_dist=100)

    if image is not None: 
        SatelliteDB.at[r, ['image']] = [postP(image).tolist()] #[postP(image)]
        SatelliteDB.loc[r, ['cloud_cover', 'vegetation_percentage', 'swir_min', 'swir_max', 'swir_mean', 'generation_time', 'nimages']] = [cloud_cover, vegetation_percentage, swir_min, swir_max, swir_mean, generation_time, nimages]
    else: 
        SatelliteDB.at[r, ['image']] = ["-"] #[postP(image)]
        SatelliteDB.loc[r, ['cloud_cover', 'vegetation_percentage', 'swir_min', 'swir_max', 'swir_mean', 'generation_time', 'nimages']] = ["-", "-", "-", "-", "-", "-", "-"]
    
    if r%50 == 0:# Save the df
        SatelliteDB.to_csv("Data/df_satellite.csv", sep = ";")
            
            
            