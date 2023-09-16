import pandas as pd
import numpy as np
import pytz
from datetime import datetime
    
"""------------ SETTINGS -------------"""
WEATHER_SOURCE = "OpenWeather" #Wunderground/OpenWeather
"""-----------------------------------"""

if WEATHER_SOURCE == "OpenWeather":
    # df_weather is already present (bought from openweather website)
    df_weather = pd.read_table('Data/df_weather_OD_RAW.csv', sep = ",")
    
    # Load the data (this data contains all the roads)
    df = pd.read_table('Data/df_prepared_1.csv', sep=";", index_col=0, low_memory=False)
    
    df_weather = df_weather.rename(columns={'dt': 'dt_UTC', 'dt_iso': 'dt_iso_UTC'})
    df_weather['dt_iso_UTC'] = df_weather['dt_iso_UTC'].apply(lambda x: datetime.strptime(x[0:19], '%Y-%m-%d %H:%M:%S')).dt.tz_localize('UTC')
    df_weather['dt_iso_AsiaJakarta'] = df_weather['dt_iso_UTC'].dt.tz_convert(pytz.timezone('Asia/Jakarta')) # convert the datetime column to Asia/Jakarta timezone
    # Write in desired format
    df_weather['dt_iso_UTC'] = df_weather['dt_iso_UTC'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_weather['dt_iso_AsiaJakarta'] = df_weather['dt_iso_AsiaJakarta'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Only look for dates that are actually relevant
    DatesUsed = sorted(set(list(df['DateDriven_r'])))
    DatesUsed.remove("-")
    df_weather_relevant = df_weather[df_weather['dt_iso_AsiaJakarta'].astype(str).isin(DatesUsed)].reset_index(drop = True)
    df_weather_relevant.fillna(0, inplace = True)
    
    # Add some additional features that we may want to use later on
    df_weather_relevant['rain_3h'] = df_weather_relevant.rain_1h + df_weather_relevant.rain_1h.shift(1) + df_weather_relevant.rain_1h.shift(2)
    df_weather_relevant['rain_5h'] = df_weather_relevant.rain_3h + df_weather_relevant.rain_1h.shift(3) + df_weather_relevant.rain_1h.shift(4) 
    df_weather_relevant['rain_10h'] = df_weather_relevant.rain_5h + df_weather_relevant.rain_1h.shift(5) + df_weather_relevant.rain_1h.shift(6) + df_weather_relevant.rain_1h.shift(7) + df_weather_relevant.rain_1h.shift(8) + df_weather_relevant.rain_1h.shift(9) 
    df_weather_relevant['rain_24h'] = df_weather_relevant.rain_10h + df_weather_relevant.rain_1h.shift(10) + df_weather_relevant.rain_1h.shift(11) + df_weather_relevant.rain_1h.shift(12) + df_weather_relevant.rain_1h.shift(13) + df_weather_relevant.rain_1h.shift(14) + df_weather_relevant.rain_1h.shift(15) + df_weather_relevant.rain_1h.shift(16) + df_weather_relevant.rain_1h.shift(17) + df_weather_relevant.rain_1h.shift(18) + df_weather_relevant.rain_1h.shift(19) + df_weather_relevant.rain_1h.shift(20)+df_weather_relevant.rain_1h.shift(21) + df_weather_relevant.rain_1h.shift(22) + df_weather_relevant.rain_1h.shift(23)   
    df_weather_relevant = df_weather_relevant.round({'rain_3h': 2, 'rain_5h': 2,'rain_10h': 2,'rain_24h': 2})
    
    relevant_features = ['dt_iso_AsiaJakarta', 'lat', 'lon', 'temp', 'visibility', 'dew_point', 'feels_like', 'temp_min', 'temp_max', 'pressure', 'sea_level', 'grnd_level', 'humidity', 'wind_speed', 'wind_deg', 'wind_gust', 'rain_1h', 'rain_3h', 'rain_5h','rain_10h','rain_24h', 'snow_1h', 'snow_3h', 'clouds_all', 'weather_main', 'weather_description']
    df_weather_relevant = df_weather_relevant.loc[:,relevant_features]
    
    print(df_weather_relevant)
    df_weather_relevant.to_csv('Data/df_weather_OD.csv', sep = ";")
elif WEATHER_SOURCE == "Wunderground":
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.service import Service
    def get_weather(STATIONID, DATETIME):
        """
            Extract the weather information of [STATIONID] on datetime [DATETIME]
            uses scraping the Wunderground website

        Input:
            STATIONID : String
                The stationid for which you would like to know the weather.
                (e.g., 'id/pekanbaru/WIBB')
            DATETIME : String
                The datetime for which you would like to know the weather. 
                (e.g., '2022-1-31 14:00:30')

        Returns:
            df.loc : DataFrame
                row of a dataframe that contains all the weather information
        """
        # Preprocess the input
        DATE, TIME = DATETIME.split(" ")
        date_time_obj = datetime.strptime(DATETIME, '%Y-%m-%d %H:%M:%S')
        url = 'https://www.wunderground.com/history/daily/'+STATIONID+'/date/'+DATE

        # Set up the web driver (using selenium)
        opt = webdriver.ChromeOptions()
        opt.add_argument('headless') # do not open chrome explicitly
        opt.add_argument('--incognito')
        chrome_driver = webdriver.Chrome(service=Service("./chromedriver"), options = opt)
        chrome_driver.get(url) # Set up the chrome driver with the url
        # Get the response, wait for the table to load. 
        WebDriverWait(chrome_driver, 60).until(EC.visibility_of_element_located((By.XPATH, '//*[@id="inner-content"]/div[2]/div[1]/div[5]/div[1]/div/lib-city-history-observation/div/div[2]/table/tbody')))
        r = chrome_driver.find_element(By.XPATH,'//*[@id="inner-content"]/div[2]/div[1]/div[5]/div[1]/div/lib-city-history-observation/div/div[2]/table/tbody')
        # Set up the dataframe (that will contain the table)
        df = pd.DataFrame([x.split(' ') for x in r.text.split('\n')])
        # Find last column and concatenate all the separate words
        s = df.iloc[:,17].map(str)
        for i in range(18, df.shape[1]):
            s +=  " " + df.iloc[:,i].fillna(value="", inplace=False).map(str)  
        df.iloc[:,17] = s 
        # Combine the first two columns into a single column (representing the time of the day)
        df.iloc[:,0] = df.iloc[:,0].map(str) + " " + df.iloc[:,1].fillna(value="", inplace=False).map(str) 
        # Remove all units of measurements
        df = df.iloc[:,[0,2,4,6,8,9,11,13,15,17]] 
        df.columns = ['Time_Used', 'Temperature (°C)', 'Dew_Point (°F)', 'Humidity (%)', 'Wind', 'Wind Speed (mph)', 'Wind Gust (mph)', 'Atmospheric Pressure (in from mercury)', 'Precipitation (in)', 'Condition']
        df['Condition'] = df['Condition'].str.rstrip()
        df['Temperature (°C)'] = (df['Temperature (°C)'].map(int) - 32)*(5/9)
        df['Temperature (°C)'] = df['Temperature (°C)'].astype(int)
        df['Time_Used'] = pd.to_datetime(DATE + " " +df['Time_Used'])
        # Return the weather (the output)
        return df.loc[df.Time_Used == min(df['Time_Used'], key=lambda x: abs(x - date_time_obj)),:]

    # Test the get_weather function
    # a = get_weather(STATIONID = 'id/pekanbaru/WIBB', DATETIME = '2022-1-31 14:00:30') #'id/pekanbaru/WIBB' #'is/reykjav%C3%ADk/BIRK' #'nl/eindhoven/EHEH'
    # print(a)

    # Load the data (this data contains all the roads)
    df = pd.read_table('Data/df_prepared_1.csv', sep=";", index_col=0, low_memory = False)
    # Only look for dates that are actually relevant
    DatesUsed = sorted(set(list(df['DateDriven_r'])))
    DatesUsed.remove("-")
    
    # Create weather df    
    df_weather = pd.DataFrame(columns = ['dt_iso_AsiaJakarta', 'Temperature (°C)', 'Dew_Point (°F)', 'Humidity (%)', 'Wind', 'Wind Speed (mph)', 'Wind Gust (mph)', 'Atmospheric Pressure (in from mercury)', 'Precipitation (in)', 'Condition'])
    df_weather.dt_iso_AsiaJakarta = DatesUsed

    DATE = ""
    for i in range(0,len(df_weather)):
        print(i)
        if (str(df_weather.dt_iso_AsiaJakarta[i]).split(" ")[0] == DATE) & (i>0):
            # Date is not available continue
            print("Same date, also not available...")
            continue
        try:
            a = get_weather(STATIONID = 'id/pekanbaru/WIBB', DATETIME = str(df_weather.dt_iso_AsiaJakarta[i])).reset_index(drop=True)
            df_weather.loc[i, df_weather.columns != 'dt_iso_AsiaJakarta'] = a.loc[0,:]
        except:
            DATE, _ = str(df_weather.dt_iso_AsiaJakarta[i]).split(" ")
            url = 'https://www.wunderground.com/history/daily/id/pekanbaru/WIBB/date/'+DATE
            print("Data not found: " + str(url))

        # Save the weather data while running
        if (i % 10)==0: df_weather.to_csv('Data/df_weather_WU.csv', sep = ";")
    print(df_weather)
    df_weather.to_csv('Data/df_weather_WU.csv', sep = ";")        