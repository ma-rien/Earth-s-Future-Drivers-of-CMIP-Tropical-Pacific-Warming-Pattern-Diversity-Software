
#!/bin/bash

# This script preprocesses CMIP .nc files for climate data analysis
# It selects specific time periods and performs grid interpolation for each model

# Ask the user to enter the variable name
echo "Enter the variable name to process (for example: rlds, hfls, hfss, etc.):"
read var

echo "Processing variable: $var"

# Navigate to the variable directory
cd "$var" || { echo "Directory $var does not exist. Script terminated."; exit 1; }
cd interpolated/masked

# Loop through all CMIP5 and CMIP6 models
# This list contains 63 climate models used in the analysis
for mod in ACCESS1-0 ACCESS1-3 ACCESS1-CM2 ACCESS1-ESM1.5 AWI-CM-1-1-MR BCC-CSM1-1 BCC-CSM1-1-M CAMS-CSM1-0 CAS-ESM2-0 CCSM4 CESM2 CESM2-WACCM CIESM CMCC-CESM CMCC-CM CMCC-CMS CMCC-CM2-SR5 CMCC-ESM2 CNRM-CM5 CNRM-CM6-1 CNRM-CM6-1-HR CNRM-ESM2-1 CSIRO-Mk3-6-0 CanESM2 CanESM5 E3SM-1-1 EC-Earth3 EC-Earth3-CC EC-Earth3-Veg EC-Earth3-Veg-LR FGOALS-f3-L FGOALS-g3 FIO-ESM-2-0 GFDL-CM3 GFDL-CM4 GFDL-ESM2G GFDL-ESM2M GFDL-ESM4 INM-CM4 IPSL-CM5A-LR IPSL-CM5A-MR IPSL-CM5B-LR IPSL-CM6A-LR KIOST-ESM MIROC5 MIROC6 MIROC-ESM MIROC-ESM-CHEM MIROC-ES2L MPI-ESM-LR MPI-ESM-MR MPI-ESM1-2-HR MPI-ESM1-2-LR MRI-CGCM3 MRI-ESM1 MRI-ESM2-0 NESM3 NorESM1-M NorESM1-ME NorESM2-LM NorESM2-MM TaiESM1 UKESM1-0-LL
do
    echo "Processing model: ${mod}"
    
    # Extract and process historical period (1900-1925)
    # This command:
    # 1. Selects years 1900-1925 from the masked file
    # 2. Calculates the time mean
    # 3. Remaps to a common grid using distance-weighted averaging
    cdo -remapdis,common_grille.nc -timmean -selyear,1900/1925 file_${var}_${mod}.nc_masked.nc timmean_file_${var}_${mod}_1900_1925.nc
    
    # Extract and process future period (2075-2100)
    # Similar to above but for years 2075-2100
    cdo -remapdis,common_grille.nc -timmean -selyear,2075/2100 file_${var}_${mod}.nc_masked.nc timmean_file_${var}_${mod}_2075_2100.nc
done

# Create directories for output organization
mkdir -p 1900_1925 
mkdir -p 2075_2100 

# Move processed files to their respective directories
mv *1900_1925* 1900_1925/
mv *2075_2100* 2075_2100/

# Return to the initial directory
cd ../../..

echo "Processing completed for variable $var."
