# MLTSA Project Data

This repository contains the following data files:
 * `PJM_NEPOOL_hourly_data_Jan2000_Jul2019.xlsx`

    The primary dataset used in this project is the Net energy peak and load by source for NEPOOL. In this dataset, load is the amount of energy that is consumed at a single point, expressed in Megawatt-hours. This dataset covers the time interval from January 1st, 2000 to July 29th, 2019. Figure 1 shows a typical energy demand for the year 2015. The other relevant feature in this dataset is the Heat Index, which combines historical temperature and humidity data at Boston Logan International Airport, measured in degrees Fahrenheit. Figure 2 shows the variation in energy consumption for the area for the period of 50 days starting on 21st of February, 2001.
    
 * `boston_weather.csv`

    The second dataset includes the historical weather data for the Boston area. This data was acquired from Climate Data Online (CDO), presented by NOAA. The full dataset contained many different attributes relating to the hourly weather conditions over the time period covered by the NEPOOL dataset.

 * `boston_weather_hourly_1.csv`

    Due to limits on the size of datasets that can be ordered at a time, two separate datasets were requested and then concatenated together. This is the first half (chronologically) of the full dataset.

 * `boston_weather_hourly_2.csv`

    Due to limits on the size of datasets that can be ordered at a time, two separate datasets were requested and then concatenated together. This is the second half (chronologically) of the full dataset.

 * `data.csv`

    This is a reduced (and combined) version of the NEPOOL and NOAA datasets. Only three features are present in this dataset: 
    * Timestamp
    * Energy load
    * First principal component of the NOAA features
 
 * `data_st.csv`

    This is a standardized version of `data.csv`. The timestamp is not included in this dataset.

 * `trained_model_1.pickle`

    This is a preliminary trainded model created using Facebook's Prophet.

