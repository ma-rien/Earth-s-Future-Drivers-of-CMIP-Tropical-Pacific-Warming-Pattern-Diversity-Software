#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 12:21:41 2025

@author: vincent
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 20:53:46 2025

@author: vincent
"""

# CMIPS Data Processing
# This script processes climate model data for different regions (WEST, EAST, SOUTH, EQUATOR)
# and calculates various statistical measures

# WEST REGION PROCESSING
# ----------------------

# Calculate relative temperature change for WEST region
relative_tos_CMIPS_WEST_change_list = []
for i in range(n):
    # Extract data for WEST region (lat: 78:82, lon: 150:200)
    test = relative_tos_change_array[i,78:82,150:200]
    relative_tos_CMIPS_WEST_change_list.append(test)  
# Convert list to numpy array
relative_tos_CMIPS_WEST_change_array = np.array(relative_tos_CMIPS_WEST_change_list)
    
# Calculate zonal mean (average along latitude)
relative_tos_CMIPS_WEST_change_zonal = np.nanmean(relative_tos_CMIPS_WEST_change_array,axis=1)   
# Calculate overall mean
relative_tos_CMIPS_WEST_change_mean = np.nanmean(relative_tos_CMIPS_WEST_change_zonal,axis=1)   

# Process atmospheric variables for WEST region
Patterns_atm_variable_total_CMIPS_WEST_change_list = []
# Loop through all atmospheric variables
for j in range(len(Patterns_atm_variable_total_array)):
    atm_var = Patterns_atm_variable_total_array[j]
    relative_atm_variable_CMIPS_WEST_change_list = [] 
    
    # Extract data for each time step
    for i in range(n):
        test = atm_var[i,78:82,150:200]
        relative_atm_variable_CMIPS_WEST_change_list.append(test)
    
    Patterns_atm_variable_total_CMIPS_WEST_change_list.append(relative_atm_variable_CMIPS_WEST_change_list)

# Convert lists to arrays and calculate means
Patterns_atm_variable_total_CMIPS_WEST_change_array = np.array(Patterns_atm_variable_total_CMIPS_WEST_change_list)  
Patterns_atm_variable_total_CMIPS_WEST_change_array_zonal = np.nanmean(Patterns_atm_variable_total_CMIPS_WEST_change_list,axis=2)
Patterns_atm_variable_total_CMIPS_WEST_change_array_mean = np.nanmean(Patterns_atm_variable_total_CMIPS_WEST_change_array_zonal,axis=2)

# Process atmospheric component patterns for WEST region
Pattern_sum_atm_component_fraction_CMIPS_WEST_change_list = []
for i in range(n):
    test = Pattern_atm_somme_component_change[i,78:82,150:200]
    Pattern_sum_atm_component_fraction_CMIPS_WEST_change_list.append(test)  
Pattern_sum_atm_component_fraction_CMIPS_WEST_change_array = np.array(Pattern_sum_atm_component_fraction_CMIPS_WEST_change_list)

# Calculate zonal and total means
Pattern_sum_atm_component_fraction_CMIPS_WEST_change_zonal = np.nanmean(Pattern_sum_atm_component_fraction_CMIPS_WEST_change_array,axis=1)   
Pattern_sum_atm_component_fraction_CMIPS_WEST_change_mean = np.nanmean(Pattern_sum_atm_component_fraction_CMIPS_WEST_change_zonal,axis=1)   

# Similar pattern continues for other variables and regions (EAST, SOUTH, EQUATOR)
# Each section follows the same structure:
# 1. Extract regional data
# 2. Convert to numpy array
# 3. Calculate zonal means
# 4. Calculate overall means

# Note: The process is repeated for each region with different latitude/longitude ranges:
# WEST:    lat 78:82,  lon 150:200
# EAST:    lat 78:82,  lon 220:270
# SOUTH:   lat 57:67,  lon 230:280
# EQUATOR: lat 78:82,  lon 150:270



# Process relative D0 patterns for WEST region
Patterns_relative_D0_CMIPS_WEST_change_list = []
for i in range(n):
    # Extract D0 patterns for WEST region
    test = Patterns_relative_D0_change_array[i,78:82,150:200]
    Patterns_relative_D0_CMIPS_WEST_change_list.append(test)  
Patterns_relative_D0_CMIPS_WEST_change_array = np.array(Patterns_relative_D0_CMIPS_WEST_change_list)
    
# Calculate means for D0 patterns
Patterns_relative_D0_CMIPS_WEST_change_zonal = np.nanmean(Patterns_relative_D0_CMIPS_WEST_change_array,axis=1)   
Patterns_relative_D0_CMIPS_WEST_change_mean = np.nanmean(Patterns_relative_D0_CMIPS_WEST_change_zonal,axis=1)   

# Process oceanic flux deviation fractions for WEST region
oceanic_flux_deviation_fraction_CMIPS_WEST_change_list = []
for i in range(n):
    # Extract oceanic flux deviations
    test = oceanic_flux_deviation_array_fraction[i,78:82,150:200]
    oceanic_flux_deviation_fraction_CMIPS_WEST_change_list.append(test)  
oceanic_flux_deviation_fraction_CMIPS_WEST_change_array = np.array(oceanic_flux_deviation_fraction_CMIPS_WEST_change_list)
    
# Calculate means for oceanic flux deviations
oceanic_flux_deviation_fraction_CMIPS_WEST_change_zonal = np.nanmean(oceanic_flux_deviation_fraction_CMIPS_WEST_change_array,axis=1)   
oceanic_flux_deviation_fraction_CMIPS_WEST_change_mean = np.nanmean(oceanic_flux_deviation_fraction_CMIPS_WEST_change_zonal,axis=1)

# Process spatial heterogeneity fractions for WEST region
Spatial_heterogeneity_fraction_array_CMIPS_WEST_change_list = []
for i in range(n):
    # Extract spatial heterogeneity data
    test = Spatial_heterogeneity_fraction_array[i,78:82,150:200]
    Spatial_heterogeneity_fraction_array_CMIPS_WEST_change_list.append(test)  
Spatial_heterogeneity_fraction_array_CMIPS_WEST_change_array = np.array(Spatial_heterogeneity_fraction_array_CMIPS_WEST_change_list)
    
# Calculate means for spatial heterogeneity
Spatial_heterogeneity_fraction_array_CMIPS_WEST_change_zonal = np.nanmean(Spatial_heterogeneity_fraction_array_CMIPS_WEST_change_array,axis=1)   
Spatial_heterogeneity_fraction_array_CMIPS_WEST_change_mean = np.nanmean(Spatial_heterogeneity_fraction_array_CMIPS_WEST_change_zonal,axis=1)   

# Process RSST component sums for WEST region
RSST_somme_component_CMIPS_WEST_change_list = []
for i in range(n):
    # Extract RSST component data
    test = RSST_somme_component_change_ALPHA_MMM[i,78:82,150:200]
    RSST_somme_component_CMIPS_WEST_change_list.append(test)
    RSST_somme_component_CMIPS_WEST_change = np.array(RSST_somme_component_CMIPS_WEST_change_list)
# Calculate means for RSST components
RSST_somme_component_CMIPS_WEST_change_zonal = np.nanmean(RSST_somme_component_CMIPS_WEST_change,axis=1)   
RSST_somme_component_CMIPS_WEST_change_mean = np.nanmean(RSST_somme_component_CMIPS_WEST_change_zonal,axis=1)   

# Process oceanic component sums for WEST region
relative_sum_oceanic_component_fraction_CMIPS_WEST_change_list = []
for i in range(n):
    # Extract oceanic component data
    test = oceanic_somme_component_change[i,78:82,150:200]
    relative_sum_oceanic_component_fraction_CMIPS_WEST_change_list.append(test)  
relative_sum_oceanic_component_fraction_CMIPS_WEST_change_array = np.array(relative_sum_oceanic_component_fraction_CMIPS_WEST_change_list)
    
# Calculate means for oceanic components
relative_sum_oceanic_component_fraction_CMIPS_WEST_change_zonal = np.nanmean(relative_sum_oceanic_component_fraction_CMIPS_WEST_change_array,axis=1)   
relative_sum_oceanic_component_fraction_CMIPS_WEST_change_mean = np.nanmean(relative_sum_oceanic_component_fraction_CMIPS_WEST_change_zonal,axis=1)   

# EAST REGION PROCESSING
# ---------------------
# Similar process as WEST region but for coordinates lat: 78:82, lon: 220:270

# Process temperature changes for EAST region
relative_tos_CMIPS_EAST_change_list = []
for i in range(n):
    # Extract temperature data for EAST region
    test = relative_tos_change_array[i,78:82,220:270]
    relative_tos_CMIPS_EAST_change_list.append(test)  
relative_tos_CMIPS_EAST_change_array = np.array(relative_tos_CMIPS_EAST_change_list)


# Calculate zonal and total means for EAST temperature
relative_tos_CMIPS_EAST_change_zonal = np.nanmean(relative_tos_CMIPS_EAST_change_array,axis=1)   
relative_tos_CMIPS_EAST_change_mean = np.nanmean(relative_tos_CMIPS_EAST_change_zonal,axis=1)   

# Process atmospheric variables for EAST region
Patterns_atm_variable_total_CMIPS_EAST_change_list = []
# Loop through each atmospheric variable
for j in range(len(Patterns_atm_variable_total_array)):
    atm_var = Patterns_atm_variable_total_array[j]
    relative_atm_variable_CMIPS_EAST_change_list = [] 
    
    # Extract data for each time step in EAST region
    for i in range(n):
        test = atm_var[i,78:82,220:270]
        relative_atm_variable_CMIPS_EAST_change_list.append(test)
    
    Patterns_atm_variable_total_CMIPS_EAST_change_list.append(relative_atm_variable_CMIPS_EAST_change_list)

# Convert atmospheric data to arrays and calculate means
Patterns_atm_variable_total_CMIPS_EAST_change_array = np.array(Patterns_atm_variable_total_CMIPS_EAST_change_list)  
Patterns_atm_variable_total_CMIPS_EAST_change_array_zonal = np.nanmean(Patterns_atm_variable_total_CMIPS_EAST_change_list,axis=2)
Patterns_atm_variable_total_CMIPS_EAST_change_array_mean = np.nanmean(Patterns_atm_variable_total_CMIPS_EAST_change_array_zonal,axis=2)

# Process atmospheric component patterns for EAST region
Pattern_sum_atm_component_fraction_CMIPS_EAST_change_list = []
for i in range(n):
    # Extract atmospheric component data
    test = Pattern_atm_somme_component_change[i,78:82,220:270]
    Pattern_sum_atm_component_fraction_CMIPS_EAST_change_list.append(test)  
Pattern_sum_atm_component_fraction_CMIPS_EAST_change_array = np.array(Pattern_sum_atm_component_fraction_CMIPS_EAST_change_list)
    
# Calculate means for atmospheric components
Pattern_sum_atm_component_fraction_CMIPS_EAST_change_zonal = np.nanmean(Pattern_sum_atm_component_fraction_CMIPS_EAST_change_array,axis=1)   
Pattern_sum_atm_component_fraction_CMIPS_EAST_change_mean = np.nanmean(Pattern_sum_atm_component_fraction_CMIPS_EAST_change_zonal,axis=1)   

# Process D0 patterns for EAST region
Patterns_relative_D0_CMIPS_EAST_change_list = []
for i in range(n):
    # Extract D0 pattern data
    test = Patterns_relative_D0_change_array[i,78:82,220:270]
    Patterns_relative_D0_CMIPS_EAST_change_list.append(test)  
Patterns_relative_D0_CMIPS_EAST_change_array = np.array(Patterns_relative_D0_CMIPS_EAST_change_list)
    
# Calculate means for D0 patterns
Patterns_relative_D0_CMIPS_EAST_change_zonal = np.nanmean(Patterns_relative_D0_CMIPS_EAST_change_array,axis=1)   
Patterns_relative_D0_CMIPS_EAST_change_mean = np.nanmean(Patterns_relative_D0_CMIPS_EAST_change_zonal,axis=1)   

# Process oceanic flux deviations for EAST region
oceanic_flux_deviation_fraction_CMIPS_EAST_change_list = []
for i in range(n):
    # Extract oceanic flux data
    test = oceanic_flux_deviation_array_fraction[i,78:82,220:270]
    oceanic_flux_deviation_fraction_CMIPS_EAST_change_list.append(test)  
oceanic_flux_deviation_fraction_CMIPS_EAST_change_array = np.array(oceanic_flux_deviation_fraction_CMIPS_EAST_change_list)
    
# Calculate means for oceanic flux
oceanic_flux_deviation_fraction_CMIPS_EAST_change_zonal = np.nanmean(oceanic_flux_deviation_fraction_CMIPS_EAST_change_array,axis=1)   
oceanic_flux_deviation_fraction_CMIPS_EAST_change_mean = np.nanmean(oceanic_flux_deviation_fraction_CMIPS_EAST_change_zonal,axis=1)


# Process spatial heterogeneity for EAST region
Spatial_heterogeneity_fraction_array_CMIPS_EAST_change_list = []
for i in range(n):
    # Extract spatial heterogeneity data for EAST region
    test = Spatial_heterogeneity_fraction_array[i,78:82,220:270]
    Spatial_heterogeneity_fraction_array_CMIPS_EAST_change_list.append(test)  
Spatial_heterogeneity_fraction_array_CMIPS_EAST_change_array = np.array(Spatial_heterogeneity_fraction_array_CMIPS_EAST_change_list)
    
# Calculate means for spatial heterogeneity in EAST region
Spatial_heterogeneity_fraction_array_CMIPS_EAST_change_zonal = np.nanmean(Spatial_heterogeneity_fraction_array_CMIPS_EAST_change_array,axis=1)   
Spatial_heterogeneity_fraction_array_CMIPS_EAST_change_mean = np.nanmean(Spatial_heterogeneity_fraction_array_CMIPS_EAST_change_zonal,axis=1)   

# Process oceanic component sums for EAST region
relative_sum_oceanic_component_fraction_CMIPS_EAST_change_list = []
for i in range(n):
    # Extract oceanic component data
    test = (oceanic_somme_component_change[i,78:82,220:270])
    relative_sum_oceanic_component_fraction_CMIPS_EAST_change_list.append(test)  
relative_sum_oceanic_component_fraction_CMIPS_EAST_change_array = np.array(relative_sum_oceanic_component_fraction_CMIPS_EAST_change_list)
    
# Calculate means for oceanic components
relative_sum_oceanic_component_fraction_CMIPS_EAST_change_zonal = np.nanmean(relative_sum_oceanic_component_fraction_CMIPS_EAST_change_array,axis=1)   
relative_sum_oceanic_component_fraction_CMIPS_EAST_change_mean = np.nanmean(relative_sum_oceanic_component_fraction_CMIPS_EAST_change_zonal,axis=1)   

# Process RSST component sums for EAST region
RSST_somme_component_CMIPS_EAST_change_list = []
for i in range(n):
    # Extract RSST component data
    test = RSST_somme_component_change_ALPHA_MMM[i,78:82,220:270]
    RSST_somme_component_CMIPS_EAST_change_list.append(test)
    RSST_somme_component_CMIPS_EAST_change = np.array(RSST_somme_component_CMIPS_EAST_change_list)

# Calculate means for RSST components
RSST_somme_component_CMIPS_EAST_change_zonal = np.nanmean(RSST_somme_component_CMIPS_EAST_change,axis=1)   
RSST_somme_component_CMIPS_EAST_change_mean = np.nanmean(RSST_somme_component_CMIPS_EAST_change_zonal,axis=1)

# SOUTH REGION PROCESSING
# ----------------------
# Process temperature changes for SOUTH region (lat: 57:67, lon: 230:280)
relative_tos_CMIPS_SOUTH_change_list = []
for i in range(n):
    # Extract temperature data for SOUTH region
    test = relative_tos_change_array[i,57:67,230:280]
    relative_tos_CMIPS_SOUTH_change_list.append(test)  
relative_tos_CMIPS_SOUTH_change_array = np.array(relative_tos_CMIPS_SOUTH_change_list)
    
# Calculate zonal and total means for SOUTH temperature
relative_tos_CMIPS_SOUTH_change_zonal = np.nanmean(relative_tos_CMIPS_SOUTH_change_array,axis=1)   
relative_tos_CMIPS_SOUTH_change_mean = np.nanmean(relative_tos_CMIPS_SOUTH_change_zonal,axis=1)   

# Process atmospheric variables for SOUTH region
Patterns_atm_variable_total_CMIPS_SOUTH_change_list = []
# Loop through each atmospheric variable
for j in range(len(Patterns_atm_variable_total_array)):
    atm_var = Patterns_atm_variable_total_array[j]
    relative_atm_variable_CMIPS_SOUTH_change_list = [] 
    
    # Extract data for each time step in SOUTH region
    for i in range(n):
        test = atm_var[i,57:67,230:280]
        relative_atm_variable_CMIPS_SOUTH_change_list.append(test)
    
    Patterns_atm_variable_total_CMIPS_SOUTH_change_list.append(relative_atm_variable_CMIPS_SOUTH_change_list)

# Convert lists to arrays and calculate means for SOUTH atmospheric variables
Patterns_atm_variable_total_CMIPS_SOUTH_change_array = np.array(Patterns_atm_variable_total_CMIPS_SOUTH_change_list)  
Patterns_atm_variable_total_CMIPS_SOUTH_change_array_zonal = np.nanmean(Patterns_atm_variable_total_CMIPS_SOUTH_change_list,axis=2)
Patterns_atm_variable_total_CMIPS_SOUTH_change_array_mean = np.nanmean(Patterns_atm_variable_total_CMIPS_SOUTH_change_array_zonal,axis=2)

# EQUATOR REGION PROCESSING
# ------------------------
# Process temperature changes for EQUATOR region (lat: 78:82, lon: 150:270)
relative_tos_CMIPS_EQUATOR_change_list = []
for i in range(n):
    # Extract temperature data for EQUATOR region
    test = relative_tos_change_array[i,78:82,150:270]
    relative_tos_CMIPS_EQUATOR_change_list.append(test)  
relative_tos_CMIPS_EQUATOR_change_array = np.array(relative_tos_CMIPS_EQUATOR_change_list)
    
# Calculate zonal and total means for EQUATOR temperature
relative_tos_CMIPS_EQUATOR_change_zonal = np.nanmean(relative_tos_CMIPS_EQUATOR_change_array,axis=1)   
relative_tos_CMIPS_EQUATOR_change_mean = np.nanmean(relative_tos_CMIPS_EQUATOR_change_zonal,axis=1)   

# Process atmospheric variables for EQUATOR region
Patterns_atm_variable_total_CMIPS_EQUATOR_change_list = []
# Loop through each atmospheric variable
for j in range(len(Patterns_atm_variable_total_array)):
    atm_var = Patterns_atm_variable_total_array[j]
    relative_atm_variable_CMIPS_EQUATOR_change_list = [] 
    
    # Extract data for each time step
    for i in range(n):
        test = atm_var[i,78:82,150:270]
        relative_atm_variable_CMIPS_EQUATOR_change_list.append(test)
    
    Patterns_atm_variable_total_CMIPS_EQUATOR_change_list.append(relative_atm_variable_CMIPS_EQUATOR_change_list)

# Convert to arrays and calculate means
Patterns_atm_variable_total_CMIPS_EQUATOR_change_array = np.array(Patterns_atm_variable_total_CMIPS_EQUATOR_change_list)  
Patterns_atm_variable_total_CMIPS_EQUATOR_change_array_zonal = np.nanmean(Patterns_atm_variable_total_CMIPS_EQUATOR_change_list,axis=2)
Patterns_atm_variable_total_CMIPS_EQUATOR_change_array_mean = np.nanmean(Patterns_atm_variable_total_CMIPS_EQUATOR_change_array_zonal,axis=2)


# Process D0 patterns for EQUATOR region
Patterns_relative_D0_CMIPS_EQUATOR_change_list = []
for i in range(n):
    # Extract D0 pattern data for EQUATOR
    test = Patterns_relative_D0_change_array[i,78:82,150:270]
    Patterns_relative_D0_CMIPS_EQUATOR_change_list.append(test)  
Patterns_relative_D0_CMIPS_EQUATOR_change_array = np.array(Patterns_relative_D0_CMIPS_EQUATOR_change_list)
    
# Calculate means for D0 patterns in EQUATOR region
Patterns_relative_D0_CMIPS_EQUATOR_change_zonal = np.nanmean(Patterns_relative_D0_CMIPS_EQUATOR_change_array,axis=1)   
Patterns_relative_D0_CMIPS_EQUATOR_change_mean = np.nanmean(Patterns_relative_D0_CMIPS_EQUATOR_change_zonal,axis=1)   

# Process oceanic flux deviations for EQUATOR region
oceanic_flux_deviation_fraction_CMIPS_EQUATOR_change_list = []
for i in range(n):
    # Extract oceanic flux data
    test = oceanic_flux_deviation_array_fraction[i,78:82,150:270]
    oceanic_flux_deviation_fraction_CMIPS_EQUATOR_change_list.append(test)  
oceanic_flux_deviation_fraction_CMIPS_EQUATOR_change_array = np.array(oceanic_flux_deviation_fraction_CMIPS_EQUATOR_change_list)
    
# Calculate means for oceanic flux deviations
oceanic_flux_deviation_fraction_CMIPS_EQUATOR_change_zonal = np.nanmean(oceanic_flux_deviation_fraction_CMIPS_EQUATOR_change_array,axis=1)   
oceanic_flux_deviation_fraction_CMIPS_EQUATOR_change_mean = np.nanmean(oceanic_flux_deviation_fraction_CMIPS_EQUATOR_change_zonal,axis=1)

# Process spatial heterogeneity for EQUATOR region
Spatial_heterogeneity_fraction_array_CMIPS_EQUATOR_change_list = []
for i in range(n):
    # Extract spatial heterogeneity data
    test = Spatial_heterogeneity_fraction_array[i,78:82,150:270]
    Spatial_heterogeneity_fraction_array_CMIPS_EQUATOR_change_list.append(test)  
Spatial_heterogeneity_fraction_array_CMIPS_EQUATOR_change_array = np.array(Spatial_heterogeneity_fraction_array_CMIPS_EQUATOR_change_list)
    
# Calculate means for spatial heterogeneity
Spatial_heterogeneity_fraction_array_CMIPS_EQUATOR_change_zonal = np.nanmean(Spatial_heterogeneity_fraction_array_CMIPS_EQUATOR_change_array,axis=1)   
Spatial_heterogeneity_fraction_array_CMIPS_EQUATOR_change_mean = np.nanmean(Spatial_heterogeneity_fraction_array_CMIPS_EQUATOR_change_zonal,axis=1)   

# Process oceanic component sums for EQUATOR region
relative_sum_oceanic_component_fraction_CMIPS_EQUATOR_change_list = []
for i in range(n):
    # Extract oceanic component data
    test = oceanic_somme_component_change[i,78:82,150:270]
    relative_sum_oceanic_component_fraction_CMIPS_EQUATOR_change_list.append(test)  
relative_sum_oceanic_component_fraction_CMIPS_EQUATOR_change_array = np.array(relative_sum_oceanic_component_fraction_CMIPS_EQUATOR_change_list)
    
# Calculate means for oceanic components
relative_sum_oceanic_component_fraction_CMIPS_EQUATOR_change_zonal = np.nanmean(relative_sum_oceanic_component_fraction_CMIPS_EQUATOR_change_array,axis=1)   
relative_sum_oceanic_component_fraction_CMIPS_EQUATOR_change_mean = np.nanmean(relative_sum_oceanic_component_fraction_CMIPS_EQUATOR_change_zonal,axis=1)   

# Process RSST component sums for EQUATOR region
RSST_somme_component_CMIPS_EQUATOR_change_list = []
for i in range(n):
    # Extract RSST component data
    test = RSST_somme_component_change_ALPHA_MMM[i,78:82,150:270]
    RSST_somme_component_CMIPS_EQUATOR_change_list.append(test)
    RSST_somme_component_CMIPS_EQUATOR_change = np.array(RSST_somme_component_CMIPS_EQUATOR_change_list)

# Calculate means for RSST components
RSST_somme_component_CMIPS_EQUATOR_change_zonal = np.nanmean(RSST_somme_component_CMIPS_EQUATOR_change,axis=1)   
RSST_somme_component_CMIPS_EQUATOR_change_mean = np.nanmean(RSST_somme_component_CMIPS_EQUATOR_change_zonal,axis=1)

# Process relative atmospheric variables for SOUTH region
# ----------------------------------------------------

relative_atm_variable_total_CMIPS_SOUTH_change_list = []

# Loop through each relative atmospheric variable
for j in range(len(relative_atm_variable_array)):
    atm_var = relative_atm_variable_array[j]
    relative_atm_variable_CMIPS_SOUTH_change_list = [] 
    
    # Extract data for each time step in SOUTH region
    for i in range(n):
        test = atm_var[i,57:67,230:280]
        relative_atm_variable_CMIPS_SOUTH_change_list.append(test)
    
    # Add processed variable to the total list
    relative_atm_variable_total_CMIPS_SOUTH_change_list.append(relative_atm_variable_CMIPS_SOUTH_change_list)

# Convert to numpy arrays for calculations
relative_atm_variable_total_CMIPS_SOUTH_change_array = np.array(relative_atm_variable_total_CMIPS_SOUTH_change_list)  

# Calculate zonal means (average along longitude)
relative_atm_variable_total_CMIPS_SOUTH_change_array_zonal = np.nanmean(relative_atm_variable_total_CMIPS_SOUTH_change_list,axis=2)

# Calculate overall means for the region
relative_atm_variable_total_CMIPS_SOUTH_change_array_mean = np.nanmean(relative_atm_variable_total_CMIPS_SOUTH_change_array_zonal,axis=2)

# Continue SOUTH region processing
# ------------------------------

# Process atmospheric component patterns for SOUTH region
Pattern_sum_atm_component_fraction_CMIPS_SOUTH_change_list = []
for i in range(n):
    # Extract atmospheric component data for SOUTH region
    test = Pattern_atm_somme_component_change[i,57:67,230:280]
    Pattern_sum_atm_component_fraction_CMIPS_SOUTH_change_list.append(test)  
Pattern_sum_atm_component_fraction_CMIPS_SOUTH_change_array = np.array(Pattern_sum_atm_component_fraction_CMIPS_SOUTH_change_list)
    
# Calculate means for atmospheric components in SOUTH region
Pattern_sum_atm_component_fraction_CMIPS_SOUTH_change_zonal = np.nanmean(Pattern_sum_atm_component_fraction_CMIPS_SOUTH_change_array,axis=1)   
Pattern_sum_atm_component_fraction_CMIPS_SOUTH_change_mean = np.nanmean(Pattern_sum_atm_component_fraction_CMIPS_SOUTH_change_zonal,axis=1)   

# Process D0 patterns for SOUTH region
Patterns_relative_D0_CMIPS_SOUTH_change_list = []
for i in range(n):
    # Extract D0 pattern data
    test = Patterns_relative_D0_change_array[i,57:67,230:280]
    Patterns_relative_D0_CMIPS_SOUTH_change_list.append(test)  
Patterns_relative_D0_CMIPS_SOUTH_change_array = np.array(Patterns_relative_D0_CMIPS_SOUTH_change_list)
    
# Calculate means for D0 patterns
Patterns_relative_D0_CMIPS_SOUTH_change_zonal = np.nanmean(Patterns_relative_D0_CMIPS_SOUTH_change_array,axis=1)   
Patterns_relative_D0_CMIPS_SOUTH_change_mean = np.nanmean(Patterns_relative_D0_CMIPS_SOUTH_change_zonal,axis=1)   

# Process oceanic flux deviations for SOUTH region
oceanic_flux_deviation_fraction_CMIPS_SOUTH_change_list = []
for i in range(n):
    # Extract oceanic flux data
    test = oceanic_flux_deviation_array_fraction[i,57:67,230:280]
    oceanic_flux_deviation_fraction_CMIPS_SOUTH_change_list.append(test)  
oceanic_flux_deviation_fraction_CMIPS_SOUTH_change_array = np.array(oceanic_flux_deviation_fraction_CMIPS_SOUTH_change_list)
    
# Calculate means for oceanic flux deviations
oceanic_flux_deviation_fraction_CMIPS_SOUTH_change_zonal = np.nanmean(oceanic_flux_deviation_fraction_CMIPS_SOUTH_change_array,axis=1)   
oceanic_flux_deviation_fraction_CMIPS_SOUTH_change_mean = np.nanmean(oceanic_flux_deviation_fraction_CMIPS_SOUTH_change_zonal,axis=1)

# Process spatial heterogeneity for SOUTH region
Spatial_heterogeneity_fraction_array_CMIPS_SOUTH_change_list = []
for i in range(n):
    # Extract spatial heterogeneity data
    test = Spatial_heterogeneity_fraction_array[i,57:67,230:280]
    Spatial_heterogeneity_fraction_array_CMIPS_SOUTH_change_list.append(test)  
Spatial_heterogeneity_fraction_array_CMIPS_SOUTH_change_array = np.array(Spatial_heterogeneity_fraction_array_CMIPS_SOUTH_change_list)
    
# Calculate means for spatial heterogeneity
Spatial_heterogeneity_fraction_array_CMIPS_SOUTH_change_zonal = np.nanmean(Spatial_heterogeneity_fraction_array_CMIPS_SOUTH_change_array,axis=1)   
Spatial_heterogeneity_fraction_array_CMIPS_SOUTH_change_mean = np.nanmean(Spatial_heterogeneity_fraction_array_CMIPS_SOUTH_change_zonal,axis=1)   

# Process oceanic component sums for SOUTH region
relative_sum_oceanic_component_fraction_CMIPS_SOUTH_change_list = []
for i in range(n):
    # Extract oceanic component data
    test = oceanic_somme_component_change[i,57:67,230:280]
    relative_sum_oceanic_component_fraction_CMIPS_SOUTH_change_list.append(test)  
relative_sum_oceanic_component_fraction_CMIPS_SOUTH_change_array = np.array(relative_sum_oceanic_component_fraction_CMIPS_SOUTH_change_list)
    
# Calculate means for oceanic components
relative_sum_oceanic_component_fraction_CMIPS_SOUTH_change_zonal = np.nanmean(relative_sum_oceanic_component_fraction_CMIPS_SOUTH_change_array,axis=1)   
relative_sum_oceanic_component_fraction_CMIPS_SOUTH_change_mean = np.nanmean(relative_sum_oceanic_component_fraction_CMIPS_SOUTH_change_zonal,axis=1)   

# Process RSST component sums for SOUTH region
RSST_somme_component_CMIPS_SOUTH_change_list = []
for i in range(n):
    # Extract RSST component data
    test = RSST_somme_component_change_ALPHA_MMM[i,57:67,230:280]
    RSST_somme_component_CMIPS_SOUTH_change_list.append(test)
    RSST_somme_component_CMIPS_SOUTH_change = np.array(RSST_somme_component_CMIPS_SOUTH_change_list)

# Calculate means for RSST components
RSST_somme_component_CMIPS_SOUTH_change_zonal = np.nanmean(RSST_somme_component_CMIPS_SOUTH_change,axis=1)   
RSST_somme_component_CMIPS_SOUTH_change_mean = np.nanmean(RSST_somme_component_CMIPS_SOUTH_change_zonal,axis=1)