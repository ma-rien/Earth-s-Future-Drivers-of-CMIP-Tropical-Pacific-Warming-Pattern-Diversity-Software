README.txt - Data and Code Documentation for Figure Generation

Overview
This repository contains the Python scripts and data necessary to reproduce all figures presented in Earth's Future "Drivers of CMIP Tropical Pacific Warming Pattern Diversity". The analysis is divided into two main parts: future climate changes (Figures 1-8) and historical changes (Figures 9-12) plus the figures presented in the Supplementary Information.

Environment Setup
All figures were generated using Python. The complete list of required Python packages and their specific versions is provided in the environment.yml file. This file can be used to recreate the conda environment used for the analysis and visualization components of this study.

Data preprocessing was performed using Climate Data Operators (CDO). The specific CDO version and dependencies used are documented in the cdo_environment.yml file. Both environment files are provided to ensure complete reproducibility of the workflow from raw data processing to final figure generation.
Data Preprocessing
Input data consists of NetCDF (.nc) files obtained from:

CMIP data: https://esgf-node.llnl.gov/projects/esgf-llnl/
Observational data (COBE SST): https://www.esrl.noaa.gov/psd/data/gridded/

Prior to running the analysis scripts, the NetCDF files must be preprocessed to calculate temporal averages for the following periods:

Future analysis: 1900-1925 and 2075-2100
Historical analysis: 1980-2000 and 2000-2020

For this purpose, the date_selection.sh file can be used to interpolate files onto a common grid (common_grille.nc) and select the desired temporal averages.

This preprocessing can be performed using Climate Data Operators (CDO).

Script Description and Workflow
Future Changes Analysis (Figures 1-8)

load_data.py: Loads preprocessed physical variables for each model and time period
calculations.py: Implements the methodology detailed in the manuscript
Region_extraction.py: Extracts and calculates spatial averages for the WEP, EEP, EEP-WEP, and SEP regions
figure_list.py: Generates Figures 1-8

Historical Changes Analysis (Figures 9-12)

load_data_histo.py: Loads historical data
calculations_histo.py: Applies the same methodology to historical periods
Region_extraction.py: Performs regional analysis (shared with future analysis)
figure_histo_list.py: Generates Figures 9-12

Additional Data
The file future_temperature_ranking.csv contains future ΔT' changes averaged over the EEP region for 63 CMIP models, ordered according to the descending values of Δ_hist T' changes in the same region. This dataset is used to generate Figure 11.
