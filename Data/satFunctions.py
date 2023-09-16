import ee 
import pandas as pd
import folium
import math
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
import timeit

# Trigger the authentication flow
#ee.Authenticate()

# Initialize library
ee.Initialize()

################################################################
########################### FUNCTIONS ##########################
################################################################
# Add layers to a folium map
def add_ee_layer(self, ee_image_object, vis_params, name):
    """
     This function makes it possible to add layers to a folium map. The final line
     adds it to the set of functions available to a folium map. 
    """
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles = map_id_dict['tile_fetcher'].url_format, 
        attr = 'Map Data &copy; <a href "https://earthengine.google.com/"> Google Earth Engine </a>', 
        name = name, 
        overlay = True, 
        control = True,
        opacity=1
    ).add_to(self)
# Add this function to the Folium Map object. 
folium.Map.add_ee_layer = add_ee_layer
def plotImages(images, labels = None):
    """
     This procedure can be used to visualize a specific satellite image,
     using the df_satellite dataset (that contains the images themselves)
     a maximum of 40 images can be plotted at the same time
    """
    # Plot max 40 images
    number_figures = len(images)
        
    # Determine widht,height
    w = int(min(number_figures, 8))
    h = int(1 + np.floor((number_figures-1) / 8))
        
    plt.figure()
    for i in range(0,min(len(images), 40)):
        plt.subplot(h,w,i+1)
        plt.xticks([])
        plt.yticks([])
        if labels is not None:
            plt.title(str(labels[i]))#df_satellite.highway[i])
        plt.grid(False)
        plt.imshow(images[i])
    plt.show()

def get_sentinel_image(point, radius, visualize = False, vis_name = None):
    # Define geometry
    lon, lat = point.x, point.y
    geom = ee.Geometry.Point([lon, lat]).buffer(radius).bounds()

    # Harmonized Sentinel-2 MSI: MultiSpectral Instrument, Level-2A # '2020-05-31') \
    sd = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate('2019-06-19', '2020-05-31') \
        .filterBounds(geom) \
        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 50) \
        .map(lambda image: image.clip(geom)) 
    
    # Print the number of images in the collection
    print('Number of images in collection:', sd.size().getInfo());
    #print(sd.first().getInfo()['properties']['VEGETATION_PERCENTAGE'])
    #print(ee.Image(sd.toList(sd.size()).get(1)).getInfo()['properties']['VEGETATION_PERCENTAGE'])
    
    # Define a function to calculate the mean cloud cover from the Cloud Probability Map
    def addCloudCover(image):
        cloudProbabilityMap = image.select('MSK_CLDPRB')
        cloudCover = cloudProbabilityMap.reduceRegion(
            reducer=ee.Reducer.mean(),
            scale=20
        ).get('MSK_CLDPRB')
        return image.set('cloudCover', cloudCover)   
    sd_sorted = sd.map(addCloudCover).sort('cloudCover')
    
    # Print a list of the cloudCover values
    #cloudCover_list = sd_sorted.aggregate_array('cloudCover').flatten().getInfo()
    #print('Cloud cover values:', len(cloudCover_list))

    try:
        shp = (0, 0, 0)
        im = 0
        while (shp[0]<(radius*2/10)) or (shp[1]<(radius*2/10)):
            # Select the image with the least amount of cloudcover
            image = ee.Image(sd_sorted.toList(sd_sorted.size()).get(im)) #sd_sorted.first()
            
            # If cloudCover could not be found, we do not use the image, too risky.
            properties = image.getInfo()['properties']
            if "cloudCover" not in properties: 
                im+=1
                continue
            
            band_dict = image.sampleRectangle(defaultValue = 10).select(['B12', 'TCI_R', 'TCI_G', 'TCI_B']).getInfo()['properties']
            R, G, B, SWIR2 = band_dict['TCI_R'], band_dict['TCI_G'], band_dict['TCI_B'], band_dict['B12']
            image_array = np.array([R, G, B]).transpose((1, 2, 0))
            shp = image_array.shape
            im += 1
    except:
        print("No picture available in specified time frame..")
        return None, None, None, None, None, None, None

    # Compare the cloudcover and CLOUDY_PIXEL_PERCENTAGE
    # print(image.get('cloudCover').getInfo())
    # print(image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo())
    
    ################ Extract properties we want to include in our analysis ###############
    # (note that swir followed from bands (not properties)) If one is not available, #####
    # just ignore the value. note that we require the cloudCover to be present! See above...
    cloud_cover = properties['cloudCover']
    generation_time = properties.get('GENERATION_TIME', '-')
    vegetation_percentage = properties.get('VEGETATION_PERCENTAGE', '-')
    if not np.isnan(SWIR2).any():
        swir_min, swir_max, swir_mean = np.min(SWIR2), np.max(SWIR2), np.mean(SWIR2)
    else:
        swir_min, swir_max, swir_mean = "-", "-", "-"
    ################################################################################
    
    if visualize:
        # Visualize the image itself
        plotImages([image_array])
        # Visualize the satellite photo
        map_18 = folium.Map(location = [lat, lon], zoom_start = 15)
        visualization = {'min': 0,'max': 255,'bands': ['TCI_R', 'TCI_G', 'TCI_B']}
        map_18.add_ee_layer(image, vis_params = visualization, name = 'RGB')
        map_18.save("Figures/"+vis_name+'.html')
    
    return image_array, cloud_cover, generation_time, vegetation_percentage, swir_min, swir_max, swir_mean
################################################################
################################################################

""" --------------------------------------------------------"""
"""This file can also be used to test different satellite sources. 
To do this, you need to run this file, after specifying the settings."""
""" ---------------------------------------------------------"""
if __name__ == "__main__":
    """------------ Settings ------------"""
    SENTINEL = True
    LC_MODIS = False
    ELEVATION = False
    RAIN = False
    """----------------------------------"""

    """ -------------------- SENTINEL ---------------------- """
    if SENTINEL:
        # lon: Between -180 (W), 180 (E) Lat: # Between -90 (S), 90 (N)
        point = Point(102.4472581549724, -0.4176948569595255)#102.41130, -0.55491)# #102.26652, -0.66043  # Home: (4.977072, 51.579640) # TEST (PEMPEM): (102.3996, -0.71)
        radius = 80 #meters
    
        t1 = timeit.Timer(lambda: get_sentinel_image(point, radius, visualize = True, vis_name = "sentinel_50"))
        print('Method 1:', t1.timeit(number=1), "seconds")
    """ ----------------------------------------------------- """

    """ --------------- LAND COVER (MODIS) ----------------- """
    if LC_MODIS:
        # The MCD12Q1 V6 product provides global land cover types at yearly intervals (2001-2016) derived from six different classification schemes. Resolution: 500 meters.
        lc = ee.ImageCollection('MODIS/006/MCD12Q1').first()#.select('LC_Type1')

        # Define the location of interest as a point near Lyon France
        lon, lat = 4.8148, 45.7758 
        poi = ee.Geometry.Point(lon, lat)
        lc_poi = lc.sample(poi, scale = 1000).first().get('LC_Type1').getInfo()
        print("Land cover value at point is:", lc_poi)

        # Visualize the data
        map_18 = folium.Map(location = [37.5010, -122.1899], zoom_start = 10)
        image_viz_params = {'min': 1.0, 'max': 17.0, 'palette': [
            '05450a', '086a10', '54a708', '78d203', '009900', 'c6b044', 'dcd159',
            'dade48', 'fbff13', 'b6ff05', '27ff87', 'c24f44', 'a5a5a5', 'ff6d4c',
            '69fff8', 'f9ffa4', '1c0dff']}
        map_18.add_ee_layer(lc, vis_params = image_viz_params, name = 'Land cover')
        map_18.save('Informative_figures/modis.html')
    """ ----------------------------------------------------- """

    """ --------------------- ELEVATION --------------------- """
    if ELEVATION:
        # USGS ground elevation image (NASA SRTM Digital Elevation 30m)
        elv = ee.Image('USGS/SRTMGL1_003')

        # Define the urban location of interest as a point near Lyon France
        lon, lat  = 4.8148, 45.7758 
        poi = ee.Geometry.Point(lon, lat)
    
        elv_poi = elv.sample(poi, scale = 1000).first().get('elevation').getInfo() #We assume that this is fine when determining elevation (does not change within a kilometer here)
        print('Ground elevation at urban point:', elv_poi, 'm.')

        # Visualize the data
        map_18 = folium.Map(location = [lon, lat], zoom_start = 10)
        image_viz_params = {'bands': ['elevation'], 'min': -10, 'max': 6500, 'gamma':  [1]}
        map_18.add_ee_layer(elv, vis_params = image_viz_params, name = 'false color composite')
        map_18.save('Informative_figures/elevation.html')
    """ ----------------------------------------------------- """

    """ --------------------- RAIN DATA --------------------- """
    if RAIN:
        # Global Satellite Mapping of Precipitation (GSMaP) provides a global hourly rain rate with a 0.1 x 0.1 degree resolution.
        #rain = ee.ImageCollection('JAXA/GPM_L3/GSMaP/v6/operational').filter(ee.Filter.date('2021-03-06', '2021-03-07')).first()
        #precipitation = rain.select('hourlyPrecipRate')

        # Precipitation Estimation From Remotely Sensed Information Using Artificial Neural Networks-Climate Data Record. Resolution: 27830 meters
        rain = ee.ImageCollection('NOAA/PERSIANN-CDR').filter(ee.Filter.date('2021-03-06', '2021-03-07')).first()
        precipitation = rain.select('precipitation')

        # Visualize the data
        map_18 = folium.Map(location = [3.34, 113.03], zoom_start = 3)
        precipitationVis = {'min': 0.0,'max': 50.0,'palette': ['3907ff', '03fff3', '28ff25', 'fbff09', 'ff1105']}
        map_18.add_ee_layer(precipitation, vis_params = precipitationVis, name = 'Precipitation')
        map_18.save('Informative_figures/rain.html')
    """ ----------------------------------------------------- """
        