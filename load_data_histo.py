#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 12:04:10 2025

@author: vincent
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 20:41:41 2025

@author: vincent
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Climate Data Processing Module
Version corrected to exactly reproduce calculations from original script

This module handles loading and processing of climate data:
- Loads CMIP5/CMIP6 model outputs
- Processes atmospheric variables
- Calculates climate patterns and deviations
- Analyzes wind components
- Evaluates radiation variables
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import glob
import datetime
import sys

class ClimateDataProcessor:
    """
    Main class for processing climate model data
    Handles data loading, scaling and variable processing
    """
    def __init__(self, base_path="your_path"):
        self.base_path = base_path
        # Dictionary defining processing parameters for each climate variable
        # scaling_factor: multiplication factor for unit conversion
        # alt_name: alternative variable name in some models
        # dir_structure: subfolder path for data files
        self.variables = {
            'pr': {'scaling_factor': 86400, 'alt_name': None, 'dir_structure': 'pr'},
            'tos': {'scaling_factor': 1, 'alt_name': None, 'dir_structure': 'tos/new'},
            'hfls': {'scaling_factor': 1, 'alt_name': 'hflso', 'dir_structure': 'hfls'},
            'hfss': {'scaling_factor': 1, 'alt_name': 'hfsso', 'dir_structure': 'hfss'},
            'rlds': {'scaling_factor': 1, 'alt_name': None, 'dir_structure': 'rlds/interpolated'},
            'rlus': {'scaling_factor': 1, 'alt_name': None, 'dir_structure': 'rlus/out'},
            'rsntds': {'scaling_factor': 1, 'alt_name': None, 'dir_structure': 'rsntds'},
            'tauu': {'scaling_factor': 1, 'alt_name': 'tauuo', 'dir_structure': 'wind/tauu'},
            'tauv': {'scaling_factor': 1, 'alt_name': 'tauvo', 'dir_structure': 'wind/tauv'},
            'uas': {'scaling_factor': 1, 'alt_name': 'uaso', 'dir_structure': 'vent/uas'},
            'vas': {'scaling_factor': 1, 'alt_name': 'vaso', 'dir_structure': 'vent/vas'},
            'clt': {'scaling_factor': 1, 'alt_name': None, 'dir_structure': 'clt'}
        }

    def get_file_paths(self, var_name, period):
        """
        Get paths for all NetCDF files matching variable and time period
        
        Args:
            var_name: Climate variable name
            period: Time period identifier
            
        Returns:
            List of file paths
        """
        var_info = self.variables[var_name]
        dir_structure = var_info['dir_structure']
        full_path = f"{self.base_path}/{dir_structure}/{period}/*.nc"
        return sorted(glob.glob(full_path, recursive=True))

    def get_model_id(self, data_model):
        """Extract model identifier from dataset attributes"""
        try:
            if 'model_id' in data_model.attrs:
                return data_model.attrs['model_id']
            elif 'source_id' in data_model.attrs:
                return data_model.attrs['source_id']
            return "unknown_model"
        except:
            return "unknown_model"
            
    def get_cmip_version(self, data_model):
        """
        Determine if model is from CMIP5 or CMIP6
        Uses multiple methods:
        1. Check explicit version attributes
        2. Look for version indicators in model ID
        """
        try:
            # Check direct CMIP6 indicator
            if 'parent_mip_era' in data_model.attrs:
                if data_model.attrs['parent_mip_era'] == 'CMIP6':
                    return "CMIP6"
            
            # Check direct CMIP5 indicator    
            if 'project_id' in data_model.attrs:
                if data_model.attrs['project_id'] == 'CMIP5':
                    return "CMIP5"
            
            # Check model name for version indicators
            model_id = self.get_model_id(data_model)
            cmip5_indicators = ['CMIP5', 'IPSL-CM5', 'MIROC-ESM', 'GFDL-ESM2', 
                              'CNRM-CM5', 'bcc-csm1-1', 'inmcm4', 'MIROC5', 
                              'MPI-ESM-LR', 'MPI-ESM-MR', 'NorESM1']
            
            if any(indicator in model_id for indicator in cmip5_indicators):
                return "CMIP5"
            return "CMIP6"
            
        except Exception as e:
            print(f"Error classifying {model_id}: {str(e)}")
            return "Unknown"

    def process_variable(self, var_name, data_model, scaling_factor=1):
        """
        Extract and process climate variable data
        Handles alternative variable names and applies scaling
        
        Args:
            var_name: Variable to process
            data_model: xarray Dataset containing the data
            scaling_factor: Unit conversion factor
            
        Returns:
            Processed numpy array or None if variable not found
        """
        alt_name = self.variables[var_name]['alt_name']
        try:
            if var_name in data_model.variables:
                var_data = data_model.variables[var_name][0, :, :]
                if var_name == 'tos':  # Special processing for sea surface temperature
                    var_data = np.where(var_data < -5, np.nan, var_data)
            elif alt_name and alt_name in data_model.variables:
                var_data = data_model.variables[alt_name][0, :, :]
            else:
                return None
            return var_data * scaling_factor
        except Exception as e:
            print(f"Error processing {var_name}: {str(e)}")
            return None

    def process_period_data(self, var_name, period):
        """
        Process all data files for a given variable and time period
        
        Args:
            var_name: Climate variable to process
            period: Time period identifier
            
        Returns:
            Tuple of (data_arrays, model_ids, latitudes, longitudes)
        """
        paths = self.get_file_paths(var_name, period)
        if not paths:
            return [], [], []

        data_list = []
        model_ids = []
        lats = []
        lons = []
        scaling_factor = self.variables[var_name]['scaling_factor']

        for path in paths:
            try:
                # Special handling for temperature data timing
                decode_times = True if var_name == 'tos' else False
                data_model = xr.open_dataset(path, decode_times=decode_times)
                var_data = self.process_variable(var_name, data_model, scaling_factor)
                
                if var_data is not None:
                    data_list.append(var_data)
                    model_ids.append(self.get_model_id(data_model))
                    lats.append(data_model.variables['lat'][:])
                    lons.append(data_model.variables['lon'][:])
                
                data_model.close()
                    
            except Exception as e:
                print(f"Error processing file {path}: {str(e)}")
                continue

        return np.array(data_list), model_ids, lats, lons

def calculate_atmospheric_variables(climate_data):
    """
    Calculate atmospheric variables following original script methodology
    
    Processes:
    - Converts temperatures to Kelvin
    - Calculates LW radiation coefficients
    - Computes latent heat components
    - Determines D0 changes
    
    Args:
        climate_data: Dictionary containing processed climate variables
        
    Returns:
        Dictionary with atmospheric calculations
    """
    print("\n=== Calculating Atmospheric Variables ===")
    
    n = climate_data['metadata']['n']
    tos_change_array = climate_data['results']['tos']['change_array']
    tos_baseline = climate_data['results']['tos']['baseline']
    
    # Physical constants
    sigma = 5.67*10**-8  # Stefan-Boltzmann constant
    alpha = 0.06  # Thermal coefficient
    
    # Convert temperatures to Kelvin
    print("Converting to Kelvin...")
    tos_Kelvin_array = tos_baseline + 273
    
    print("Computing coefficients...")
    # Calculate LW upward radiation coefficients
    coef_LW_up_array = 4*sigma*(tos_Kelvin_array**3)
    coef_LW_up_array = np.where(np.isnan(tos_change_array), np.nan, coef_LW_up_array)
    
    # Calculate atmospheric LW upward flux
    rlus_change_array = climate_data['results']['rlus']['change_array']
    atm_LW_up_array = rlus_change_array - coef_LW_up_array * tos_change_array
    
    print("Processing latent heat...")
    # Process latent heat components
    hfls_baseline = climate_data['results']['hfls']['baseline']
    hfls_change_array = climate_data['results']['hfls']['change_array']
    
    # Calculate oceanic latent heat terms
    oceanic_latent_list = []
    oceanic_latent_coef_list = []
    for i in range(n):
        oceanic_latent = alpha * hfls_baseline[i] * tos_change_array[i]
        oceanic_latent_list.append(oceanic_latent)
        
        oceanic_latent_coef = -alpha * hfls_baseline[i]
        oceanic_latent_coef_list.append(oceanic_latent_coef)
    
    oceanic_latent_array = np.array(oceanic_latent_list)
    oceanic_latent_coef_array = np.array(oceanic_latent_coef_list)
    
    # Calculate atmospheric latent heat
    atm_latent_array = hfls_change_array - oceanic_latent_array
    
    # Compute total coefficients
    print("Computing total coefficients...")
    coef_up_array = oceanic_latent_coef_array + coef_LW_up_array
    
    print("Calculating D0...")
    # Calculate D0 (net surface heat flux)
    D0_change_array = (
        -climate_data['results']['rsntds']['change_array'] +
        climate_data['results']['rlus']['change_array'] -
        climate_data['results']['rlds']['change_array'] -
        climate_data['results']['hfls']['change_array'] -
        climate_data['results']['hfss']['change_array']
    )
    
    print("Computing zonal means...")
    # Calculate zonal means (55-104 latitude band represents tropics)
    D0_mean_lat = np.nanmean(D0_change_array[:,55:104,0:360], axis=1)
    D0_mean_state = np.nanmean(D0_mean_lat, axis=1)
    
    atm_LW_up_mean_lat = np.nanmean(atm_LW_up_array[:,55:104,0:360], axis=1)
    atm_LW_up_mean_state = np.nanmean(atm_LW_up_mean_lat, axis=1)
    
    # Compile atmospheric variables array
    atm_variable_change = np.array([
        climate_data['results']['rsntds']['change_array'],
        climate_data['results']['hfss']['change_array'],
        climate_data['results']['rlds']['change_array'],
        atm_latent_array,
        atm_LW_up_array
    ])
    
    print("Atmospheric variables computed.")
    
    return {
        'atmospheric': {
            'latent': {
                'oceanic': oceanic_latent_array,
                'oceanic_coef': oceanic_latent_coef_array,
                'atmospheric': atm_latent_array
            },
            'radiation': {
                'LW_up': {
                    'coef': coef_LW_up_array,
                    'atmospheric': atm_LW_up_array,
                    'mean_state': atm_LW_up_mean_state
                }
            },
            'D0': {
                'change': D0_change_array,
                'mean_state': D0_mean_state
            },
            'variables': atm_variable_change,
            'coef_up': coef_up_array
        }
    }

def calculate_patterns_and_deviations(climate_data, atm_results):
    """
    Calculate climate patterns and deviations from mean states
    
    Processes:
    - Filters land points 
    - Calculates relative changes
    - Computes atmospheric patterns
    - Determines spatial heterogeneity
    - Calculates Zhang SST estimation
    
    Args:
        climate_data: Dictionary with climate variables
        atm_results: Dictionary with atmospheric calculations
        
    Returns:
        Dictionary containing patterns and deviations
    """
    print("\n=== Computing Patterns and Deviations ===")
    
    n = climate_data['metadata']['n']
    tos_change_array = climate_data['results']['tos']['change_array']
    
    # Filter land points
    print("Filtering land points for SST...")
    tos_change_array_filter_land = np.where(tos_change_array == 0, np.nan, tos_change_array)
    
    # Calculate mean states
    print("Computing mean states...")
    # Tropical band (55:104 latitude indices)
    tos_change_mean_state_lat = np.nanmean(tos_change_array_filter_land[:,55:104,0:360], axis=1)
    tos_change_mean_state = np.nanmean(tos_change_mean_state_lat, axis=1)
    tos_change_mean_state_mean = np.nanmean(tos_change_mean_state)
    tos_change_array_tropic = tos_change_array[:,55:104,0:360]
    
    # Calculate relative SST patterns
    relative_tos_change_list = []
    for i in range(n):
        relative_tos_change = tos_change_array[i] - tos_change_mean_state[i]
        relative_tos_change_list.append(relative_tos_change)
    relative_tos_change_array = np.array(relative_tos_change_list)
    
# Process atmospheric variables
    atm_variable_change = atm_results['atmospheric']['variables']
    atm_variable_change_mean_state_lat = np.nanmean(atm_variable_change[:,:,55:104,0:360], axis=2)
    atm_variable_change_mean_state = np.nanmean(atm_variable_change_mean_state_lat, axis=2)
    
    # Calculate relative atmospheric patterns
    relative_atm_variable_total_list = []
    for j in range(len(atm_variable_change_mean_state)):
        relative_var = atm_variable_change[j]
        mean_state_var = atm_variable_change_mean_state[j]
        relative_atm_variable_list = []
        for i in range(n):
            relative_var_model = relative_var[i] - mean_state_var[i]
            relative_atm_variable_list.append(relative_var_model)
        relative_atm_variable_total_list.append(relative_atm_variable_list)
    relative_atm_variable_array = np.array(relative_atm_variable_total_list)
    
    # Process coefficients
    total_coef_array = atm_results['atmospheric']['coef_up']
    total_coef_array_mean_state_lat = np.nanmean(total_coef_array[:,55:104,0:360], axis=1)
    total_coef_array_mean_state = np.nanmean(total_coef_array_mean_state_lat, axis=1)
    total_coef_array_MMM = np.nanmean(total_coef_array, axis=0)
    
    # Calculate relative coefficients
    relative_total_coef_list = []
    for i in range(n):
        relative_total_coef = total_coef_array[i] - total_coef_array_mean_state[i]
        relative_total_coef_list.append(relative_total_coef)
    relative_total_coef_array = np.array(relative_total_coef_list)
    
    # Compute final atmospheric patterns
    print("Computing final atmospheric patterns...")
    Patterns_atm_variable_total_list = []
    for j in range(5):  # Process all 5 atmospheric variables
        relative_var = relative_atm_variable_array[j]
        Patterns_atm_variable_list = []
        for i in range(n):
            Patterns_atm_variable = relative_var[i]/total_coef_array_MMM
            Patterns_atm_variable_list.append(Patterns_atm_variable)
        Patterns_atm_variable_total_list.append(Patterns_atm_variable_list)
    Patterns_atm_variable_total_array = np.array(Patterns_atm_variable_total_list)
    
    # Calculate spatial heterogeneity
    print("Computing spatial heterogeneity...")
    Spatial_heterogeneity_array = relative_total_coef_array * relative_tos_change_array
    
    Spatial_heterogeneity_array_mean_state_lat = np.nanmean(Spatial_heterogeneity_array[:,55:104,0:360], axis=1)
    Spatial_heterogeneity_array_mean_state = np.nanmean(Spatial_heterogeneity_array_mean_state_lat, axis=1)
    
    # Calculate Zhang SST estimation
    print("Computing Zhang SST estimation...")
    tos_Zhang_change_array = ((climate_data['results']['rsntds']['change_array'] +
                              climate_data['results']['rlds']['change_array'] +
                              atm_results['atmospheric']['latent']['atmospheric'] +
                              climate_data['results']['hfss']['change_array'] +
                              atm_results['atmospheric']['D0']['change'] -
                              atm_results['atmospheric']['radiation']['LW_up']['atmospheric']) /
                             total_coef_array)
    
    tos_Zhang_change_mean_state_lat = np.nanmean(tos_Zhang_change_array[:,55:104,0:360], axis=1)
    tos_Zhang_change_mean_state = np.nanmean(tos_Zhang_change_mean_state_lat, axis=1)
    
    print("Patterns and deviations computed.")
    
    return {
        'patterns': {
            'atm_variables': {
                'total': Patterns_atm_variable_total_array,
                'MMM': np.nanmean(Patterns_atm_variable_total_array, axis=1)
            },
            'spatial_heterogeneity': {
                'array': Spatial_heterogeneity_array,
                'mean_state': Spatial_heterogeneity_array_mean_state
            },
            'oceanic_flux': {
                'deviation': relative_total_coef_array
            }
        },
        'relative_changes': {
            'tos': {
                'array': relative_tos_change_array,
                'MMM': np.nanmean(relative_tos_change_array, axis=0)
            },
            'atm_variables': {
                'array': relative_atm_variable_array,
                'MMM': np.nanmean(relative_atm_variable_array, axis=1)
            }
        },
        'tos_Zhang': {
            'array': tos_Zhang_change_array,
            'mean_state': tos_Zhang_change_mean_state
        },
        'coefficients': {
            'total': total_coef_array,
            'relative': relative_total_coef_array,
            'MMM': total_coef_array_MMM
        }
    }

def calculate_wind_variables(climate_data):
    """
    Calculate wind-related variables and their patterns
    
    Processes:
    - Wind stress components (tau)
    - Surface wind components
    - Wind stress magnitude
    - Normalized stress vectors
    
    Args:
        climate_data: Dictionary with climate variables
        
    Returns:
        Dictionary with wind calculations and metadata
    """
    print("\n=== Computing Wind Variables ===")
    
    n = climate_data['metadata']['n']
    
    # Check available wind models
    n_wind_uas = len(climate_data['results']['uas']['change_array'])
    n_wind_vas = len(climate_data['results']['vas']['change_array'])
    print(f"Number of uas models: {n_wind_uas}")
    print(f"Number of vas models: {n_wind_vas}")
    n_wind = min(n_wind_uas, n_wind_vas)
    print(f"Using n_wind = {n_wind} for surface wind calculations")
    
    # Process wind stress (tau) components
    print("Computing wind stress components...")
    tauu_change_array = climate_data['results']['tauu']['change_array']
    tauu_change_mean_state_lat = np.nanmean(tauu_change_array[:,55:104,0:360], axis=1)
    tauu_change_mean_state = np.nanmean(tauu_change_mean_state_lat, axis=1)
    
    relative_tauu_change_list = []
    for i in range(len(tauu_change_array)):
        relative_tauu_change = tauu_change_array[i] - tauu_change_mean_state[i]
        relative_tauu_change_list.append(relative_tauu_change)
    relative_tauu_change_array = np.array(relative_tauu_change_list)
    
    tauv_change_array = climate_data['results']['tauv']['change_array']
    tauv_change_mean_state_lat = np.nanmean(tauv_change_array[:,55:104,0:360], axis=1)
    tauv_change_mean_state = np.nanmean(tauv_change_mean_state_lat, axis=1)
    
    relative_tauv_change_list = []
    for i in range(len(tauv_change_array)):
        relative_tauv_change = tauv_change_array[i] - tauv_change_mean_state[i]
        relative_tauv_change_list.append(relative_tauv_change)
    relative_tauv_change_array = np.array(relative_tauv_change_list)
    
    # Calculate wind stress magnitude
    print("Computing wind stress magnitude...")
    wind_stress_change_list = []
    for i in range(len(tauu_change_array)):
        wind_stress = np.sqrt(tauu_change_array[i]**2 + tauv_change_array[i]**2)
        wind_stress_change_list.append(wind_stress)
    wind_stress_change_array = np.array(wind_stress_change_list)
    
    # Normalize stress components
    tauu_change_array_norm = tauu_change_array / np.sqrt(tauu_change_array**2 + tauv_change_array**2)
    tauv_change_array_norm = tauv_change_array / np.sqrt(tauu_change_array**2 + tauv_change_array**2)
    
    # Process surface wind components using n_wind
    print("Computing surface wind components...")
    uas_change_array = climate_data['results']['uas']['change_array'][:n_wind]
    uas_change_mean_state_lat = np.nanmean(uas_change_array[:,55:104,0:360], axis=1)
    uas_change_mean_state = np.nanmean(uas_change_mean_state_lat, axis=1)
    
    relative_uas_change_list = []
    for i in range(n_wind):
        relative_uas = uas_change_array[i] - uas_change_mean_state[i]
        relative_uas_change_list.append(relative_uas)
    relative_uas_change_array = np.array(relative_uas_change_list)
    
    vas_change_array = climate_data['results']['vas']['change_array'][:n_wind]
    vas_change_mean_state_lat = np.nanmean(vas_change_array[:,55:104,0:360], axis=1)
    vas_change_mean_state = np.nanmean(vas_change_mean_state_lat, axis=1)
    
    relative_vas_change_list = []
    for i in range(n_wind):
        relative_vas = vas_change_array[i] - vas_change_mean_state[i]
        relative_vas_change_list.append(relative_vas)
    relative_vas_change_array = np.array(relative_vas_change_list)
    
    print("Wind calculations completed.")
    
    return {
        'tau': {
            'u': {
                'change': tauu_change_array,
                'mean_state': tauu_change_mean_state,
                'relative': relative_tauu_change_array,
                'normalized': tauu_change_array_norm
            },
            'v': {
                'change': tauv_change_array,
                'mean_state': tauv_change_mean_state,
                'relative': relative_tauv_change_array,
                'normalized': tauv_change_array_norm
            },
            'stress': {
                'change': wind_stress_change_array,
                'MMM': np.nanmean(wind_stress_change_array, axis=0)
            }
        },
        'surface': {
            'u': {
                'change': uas_change_array,
                'mean_state': uas_change_mean_state,
                'relative': relative_uas_change_array
            },
            'v': {
                'change': vas_change_array,
                'mean_state': vas_change_mean_state,
                'relative': relative_vas_change_array
            }
        },
        'metadata': {
            'n_standard': n,
            'n_wind': n_wind,
            'n_wind_uas': n_wind_uas,
            'n_wind_vas': n_wind_vas
        }
    }

def calculate_radiation_variables(climate_data, total_coef_array_MMM):
    """
    Calculate radiation-related variables and their patterns
    
    Processes:
    - RLUS (upward longwave radiation) changes
    - Relative changes and patterns
    - Multi-model means and standard deviations
    
    Args:
        climate_data: Dictionary with climate variables
        total_coef_array_MMM: Multi-model mean of total coefficients
        
    Returns:
        Dictionary with radiation calculations
    """
    print("\n=== Computing Radiation Variables ===")
    
    n = climate_data['metadata']['n']
    rlus_change_array = climate_data['results']['rlus']['change_array']
    
    # Calculate RLUS zonal means
    print("Computing RLUS zonal means...")
    rlus_change_mean_state_lat = np.nanmean(rlus_change_array[:,55:104,0:360], axis=1)
    rlus_change_mean_state = np.nanmean(rlus_change_mean_state_lat, axis=1)
    rlus_change_mean_state_mean = np.nanmean(rlus_change_mean_state)
    rlus_change_array_tropic = rlus_change_array[:,55:104,0:360]
    
    # Calculate RLUS relative changes
    relative_rlus_change_list = []
    for i in range(n):
        relative_rlus = rlus_change_array[i] - rlus_change_mean_state[i]
        relative_rlus_change_list.append(relative_rlus)
    relative_rlus_change_array = np.array(relative_rlus_change_list)
    
    # Compute RLUS patterns
    print("Computing RLUS patterns...")
    Patterns_relative_rlus_change_list = []
    for i in range(n):
        pattern = relative_rlus_change_array[i]/total_coef_array_MMM
        Patterns_relative_rlus_change_list.append(pattern)
    Patterns_relative_rlus_change_array = np.array(Patterns_relative_rlus_change_list)
    
    return {
        'rlus': {
            'change': rlus_change_array,
            'mean_state': rlus_change_mean_state,
            'relative': {
                'change': relative_rlus_change_array,
                'MMM': np.nanmean(relative_rlus_change_array, axis=0),
                'STD': np.std(relative_rlus_change_array, axis=0)
            },
            'patterns': {
                'array': Patterns_relative_rlus_change_array,
                'MMM': np.nanmean(Patterns_relative_rlus_change_array, axis=0),
                'STD': np.std(Patterns_relative_rlus_change_array, axis=0)
            }
        }
    }

def load_cobe_sst_data(file_path):
    """
    Load and process COBE SST observational data
    
    Args:
        file_path: Path to COBE SST NetCDF file
        
    Returns:
        Tuple of (SST array, time-mean SST)
    """
    print("\n=== Loading COBE SST Data ===")
    try:
        data_COBE = xr.open_dataset(file_path)
        
        # Extract SST data
        tos_COBE = data_COBE.variables['sst'][:]
        tos_COBE_array = np.array(tos_COBE)
        
        # Calculate 1900-1925 time mean
        timmean_tos_COBE_1900_1925 = np.nanmean(tos_COBE, axis=0)
        
        print("COBE SST data loaded successfully")
        return tos_COBE_array, timmean_tos_COBE_1900_1925
    
    except Exception as e:
        print(f"Error loading COBE data: {str(e)}")
        raise

def main():
    """
    Main function to process all climate variables and calculations
    
    Workflow:
    1. Load base climate data
    2. Process atmospheric variables
    3. Calculate patterns
    4. Process wind variables
    5. Process radiation
    6. Load observational data
    """
    print(f"\n=== PROCESSING START - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    try:
        # 1. Initial data loading
        print("\nStep 1: Loading base climate data...")
        processor = ClimateDataProcessor()
        
        # Load SST for two periods
        results_tos_1980_2000 = processor.process_period_data('tos', "1980_2000")
        print(f"Models found for 1980_2000: {len(results_tos_1980_2000[1])}")
        
        results_tos_2000_2020 = processor.process_period_data('tos', "2000_2020")
        print(f"Models found for 2000_2020: {len(results_tos_2000_2020[1])}")
        
        # Build climate data dictionary
        climate_data = {
            'metadata': {
                'n': len(results_tos_1980_2000[0]),
                'lat': results_tos_1980_2000[2][0],
                'lon': results_tos_1980_2000[3][0],
                'model_ids': results_tos_1980_2000[1]
            },
            'results': {
                'tos': {
                    'change_array': results_tos_2000_2020[0] - results_tos_1980_2000[0],
                    'baseline': results_tos_1980_2000[0],
                    'future': results_tos_2000_2020[0]
                }
            }
        }
        
        # Classify CMIP5/CMIP6 models
        print("\n=== CMIP5/CMIP6 Classification ===")
        paths_2000_2020 = processor.get_file_paths('tos', "2000_2020")
        icmip5 = []
        icmip6 = []
        
        for i, path in enumerate(paths_2000_2020):
            data_model = xr.open_dataset(path, decode_times=True)
            version = processor.get_cmip_version(data_model)
            if version == "CMIP5":
                icmip5.append(i)
            elif version == "CMIP6":
                icmip6.append(i)
            data_model.close()
        
        climate_data['metadata'].update({
            'icmip5': np.array(icmip5),
            'icmip6': np.array(icmip6)
        })
        
        print(f"CMIP5 models: {len(icmip5)}")
        print(f"CMIP6 models: {len(icmip6)}")
        
        # Load additional variables
        print("\n=== Loading Additional Variables ===")
        variables = ['hfls', 'hfss', 'rlds', 'rlus', 'rsntds', 'tauu', 'tauv', 'uas', 'vas', 'clt']
        
        for var in variables:
            print(f"\nProcessing {var}...")
            results_1980_2000 = processor.process_period_data(var, "1980_2000")
            results_2000_2020 = processor.process_period_data(var, "2000_2020")
            
            if len(results_1980_2000[0]) > 0 and len(results_2000_2020[0]) > 0:
                climate_data['results'][var] = {
                    'change_array': results_2000_2020[0] - results_1980_2000[0],
                    'baseline': results_1980_2000[0],
                    'future': results_2000_2020[0]
                }
                print(f"  Models processed: {len(results_1980_2000[1])}")
            else:
                print(f"  WARNING: Missing data for {var}")
        
        # 2. Calculate atmospheric variables
        print("\nStep 2: Processing atmospheric calculations...")
        atm_results = calculate_atmospheric_variables(climate_data)
        
        # 3. Calculate patterns
        print("\nStep 3: Computing patterns...")
        patterns_results = calculate_patterns_and_deviations(climate_data, atm_results)
        total_coef_array_MMM = patterns_results['coefficients']['MMM']
        
        # 4. Process wind variables
        print("\nStep 4: Computing wind variables...")
        wind_results = calculate_wind_variables(climate_data)
        
        # 5. Process radiation variables
        print("\nStep 5: Computing radiation variables...")
        radiation_results = calculate_radiation_variables(climate_data, total_coef_array_MMM)
        
        # Compile final results
        results = {
            'climate_data': climate_data,
            'atmospheric': atm_results,
            'patterns': patterns_results,
            'wind': wind_results,
            'radiation': radiation_results,
            'metadata': {
                'processing_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'n_models': climate_data['metadata']['n'],
                'n_cmip5': len(icmip5),
                'n_cmip6': len(icmip6)
            }
        }
        
        # Load COBE SST observational data
        data_COBE_1980_2000 = xr.open_dataset("your_file_1980_2000.nc")
        tos_COBE_1980_2000 = data_COBE_1980_2000.variables['sst'][:]
        tos_COBE_1980_2000_array = np.array(tos_COBE_1980_2000[0])
        
        data_COBE_2000_2020 = xr.open_dataset("your_file_2000_2020.nc")
        tos_COBE_2000_2020 = data_COBE_2000_2020.variables['sst'][:]
        tos_COBE_2000_2020_array = np.array(tos_COBE_2000_2020[0])
        
        # Calculate COBE SST changes
        tos_COBE_change_array = tos_COBE_2000_2020_array - tos_COBE_1980_2000_array
        relative_tos_COBE_change_array = tos_COBE_change_array - np.nanmean(tos_COBE_change_array[55:104,0:360])
        
        # Add observational results
        climate_data['results']['tos'].update({
            'tos_COBE_change_array': tos_COBE_change_array,
            'relative_tos_COBE_change_array': relative_tos_COBE_change_array,
        })
        
        print(f"\n=== PROCESSING COMPLETE - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        return results
        
    except Exception as e:
        print(f"\n!!! ERROR DURING PROCESSING: {str(e)}")
        raise

if __name__ == "__main__":
    results = main()