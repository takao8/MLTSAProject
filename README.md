# MLTSAProject

## Abstract
Energy consumption forecasting is vitally important for energy companies and government agencies to properly hedge their assets and control prices.  In this project, we forecast the New England region’s NEPOOL energy consumption with Boston’s historical weather data from 2000 through 2019 using machine learning methods in an attempt to accurately predict New England energy consumption in the next 1-6 months.  In the first step we plan to implement Facebook's Prophet package using a decomposable time series model.  In the next step, we will create an alternative model with recurrent neural networks. The results from both models will be compared statistically.

## Introduction
In the United States, the transmission and wholesale of electricity is regulated by the Federal Energy Regulatory Commission (FERC).  Regions of the country that are deregulated are split into separate entities created by FERC, known as independent system operators (ISOs), which coordinates the electric grid of that region.  In this paper, we will be analyzing and forecasting energy demand in ISO New England (commonly called NEPOOL, the New England Power Pool), which encompasses Connecticut, Maine, Massachusetts, New Hampshire, Rhode Island, and Vermont.

Numerous reasons exist for the need to forecast load.  One such reason is for utility companies to supply energy to their customers--if the companies do not have enough supply to meet demand, they will have to buy the energy to meet that demand from the expensive real-time energy market.  To prevent such shortages, utility companies can buy cheap contracts that ensure a certain amount of energy will be supplied days, weeks, or months in advance.  These contracts act as a hedge for the companies, saving them the extraordinary cost to buy from the real-time market.  Therefore, it is necessary for utility companies to forecast energy consumption.  More accurate forecasting means more accurate hedges can be taken by companies to prevent major expenses and price swings in the energy market.

The forecasting problem becomes progressively more difficult as energy demand is isolated to smaller regions, as events such as outages will have a larger effect on the time series.  For this reason, it is desirable to start by producing proof of concept techniques on energy demand in larger areas.  This is our motivation for forecasting the entire New England region.

Load forecasting is a well-documented problem that many scientists have tackled. Yildiz [2] describes several modern methods in regression and machine learning that have been used to forecast load over the years, including autoregressive models, regularized linear regression, support vector regression, neural networks, and random forests. One of the first in-depth analysis of neural networks in load forecasting is explored by Park et al [2], reporting a mean absolute percent error of 2% on day-ahead forecasts.

## Data
Data

Two datasets are used in this project. The following table highlights the shape and types of data used in the datasets.
|Dataset name|URL|Number of rows|Number of valid rows (not NaN on relevant columns)|Number of columns|Number of relevant columns|Data type for each relevant column|
|---|---|---|---|---|---|---|
|nepool|https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/-/tree/net-ener-peak-load |171,562|171,562|18|2|All floats|
|weather_noaa|https://www.ncdc.noaa.gov/cdo-web/ |299,952|299,952|124|9|All floats|
**Table 1:** High level description of datasets.

The primary dataset used in this project is the Net energy peak and load by source for NEPOOL. In this dataset, load is the amount of energy that is consumed at a single point, expressed in Megawatt-hours. This dataset covers the time interval from January 1st, 2000 to July 29th, 2019. Figure 1 shows a typical energy demand for the year 2015. The other relevant feature in this dataset is the Heat Index, which combines historical temperature and humidity data at Boston Logan International Airport, measured in degrees Fahrenheit. Figure 2 shows the variation in energy consumption for the area for the period of 50 days starting on 21st of February, 2001.

![energy demand for 2015](figures/energy_demand_2015.png)

**Figure 1:** A time series sample of Energy demand for year 2015 for the New England area.

![snapshot of nepool.describe()](figures/nepool_data_description.png)

**Table 2:** Snapshot of `nepool.describe()`

The second dataset includes the historical weather data for the Boston area. This data was acquired from Climate Data Online (CDO), presented by NOAA. The data was acquired using the following criteria:
 * Weather Observation Type/Dataset: Normals Hourly
 * Date Range: 
   * 2000-01-01 00:00 to 2009-12-31 23:59
   * 2010-01-01 00:00 to 2019-07-29 23:59 
 * Search for: Stations
 * Search Term: WBAN:14739

Due to limits on the size of datasets that can be ordered at a time, two separate datasets were requested and then concatenated together. The resulting dataset contained many different attributes relating to the hourly weather conditions over the time period covered by the NEPOOL dataset. From the subset of categories, the following nine were highlighted as relevant features to be used in this analysis:
 * Altimeter Setting (inches)
 * Dew Point Temperature (°F)
 * Dry Bulb Temperature (°F)
 * Precipitation (inches)
 * Relative Humidity (%)
 * Sea Level Pressure (inches)
 * Wet Bulb Temperature (°F)
 * Wind Speed (mph)

![Correlation between energy consumption and Heat Index of Boston for the period between 2001-02-21and 2002-04-13](figures/hid_energy_usage.png)

**Figure 2:** Correlation between energy consumption and Heat Index of Boston for the period between 2001-02-21and 2002-04-13

![Snapshot of weather_noaa.describe()](figures/noaa_data_description.png)
**Table 3:** Snapshot of `weather_noaa.describe()`