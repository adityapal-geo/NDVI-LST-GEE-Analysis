# NDVI-LST-GEE-Analysis
Time-series NDVI &amp; LST analysis using Landsat data in Google Earth Engine


# NDVI & Land Surface Temperature (LST) Analysis using Google Earth Engine

## Overview

This project analyzes the relationship between vegetation (NDVI) and land surface temperature (LST) over time using Landsat satellite data.

## Data Used

* Landsat 5 (2004–2011)
* Landsat 8 (2013–2025)
* Region: Durgapur (India)

## Methods

* Cloud masking using QA_PIXEL
* NDVI calculation
* Emissivity estimation from NDVI
* LST calculation using radiative transfer equation
* Temporal visualization (2004–2025)

## Outputs

* NDVI vs LST correlation plot
* Time-series animation (GIF & MP4)

## Key Insight

* Negative correlation between NDVI and LST
  (Higher vegetation → Lower temperature)


## Tools & Libraries

* Google Earth Engine (GEE)
* Python
* NumPy
* Matplotlib

## How to Run

1. Open in Google Colab
2. Authenticate GEE
3. Run all cells

## Author

Aditya Pal
