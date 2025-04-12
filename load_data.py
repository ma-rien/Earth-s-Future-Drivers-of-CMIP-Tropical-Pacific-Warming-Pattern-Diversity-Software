#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 11:39:07 2025

@author: vincent
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 21:48:45 2025

@author: vincent
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 23:40:03 2025

@author: vincent
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Climate data loading and processing module
Exactly reproduces the calculations from the original script
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import glob
import datetime
import sys

class ClimateDataProcessor:
    def __init__(self, base_path="your_path"):
        self.base_path = base_path
        self.variables = {
            'pr': {'scaling_factor': 86400, 'alt_name': None, 'dir_structure': 'pr'},
            # 'tos': {'scaling_factor': 1, 'alt_name': None, 'dir_structure': 'tos/new'},
            'tos': {'scaling_factor': 1, 'alt_name': None, 'dir_structure': 'tos/new/'},

            'hfls': {'scaling_factor': 1, 'alt_name': 'hflso', 'dir_structure': 'hfls'},
            'hfss': {'scaling_factor': 1, 'alt_name': 'hfsso', 'dir_structure': 'hfss'},
            'rlds': {'scaling_factor': 1, 'alt_name': None, 'dir_structure': 'rlds/interpolated'},
            'rlus': {'scaling_factor': 1, 'alt_name': None, 'dir_structure': 'rlus/out'},
            'rsntds': {'scaling_factor': 1, 'alt_name': None, 'dir_structure': 'rsntds'},
            'tauu': {'scaling_factor': 1, 'alt_name': 'tauuo', 'dir_structure': 'wind/tauu'},
            'tauv': {'scaling_factor': 1, 'alt_name': 'tauvo', 'dir_structure': 'wind/tauv'},
            'uas': {'scaling_factor': 1, 'alt_name': 'None', 'dir_structure': 'vent/uas'},
            'vas': {'scaling_factor': 1, 'alt_name': 'None', 'dir_structure': 'vent/vas'},
            'clt': {'scaling_factor': 1, 'alt_name': None, 'dir_structure': 'clt'}
        }

    def get_file_paths(self, var_name, period):
        var_info = self.variables[var_name]
        dir_structure = var_info['dir_structure']
        full_path = f"{self.base_path}/{dir_structure}/{period}/*.nc"
        return sorted(glob.glob(full_path, recursive=True))

    def get_model_id(self, data_model):
        try:
            if 'model_id' in data_model.attrs:
                return data_model.attrs['model_id']
            elif 'source_id' in data_model.attrs:
                return data_model.attrs['source_id']
            return "unknown_model"
        except:
            return "unknown_model"
            
    def get_cmip_version(self, data_model):
        try:
            if 'parent_mip_era' in data_model.attrs and data_model.attrs['parent_mip_era'] == 'CMIP6':
                return "CMIP6"
            if 'project_id' in data_model.attrs and data_model.attrs['project_id'] == 'CMIP5':
                return "CMIP5"
                
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
        alt_name = self.variables[var_name]['alt_name']
        try:
            if var_name in data_model.variables:
                var_data = data_model.variables[var_name][0, :, :]
                if var_name == 'tos':
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
                decode_times = var_name == 'tos'
                data_model = xr.open_dataset(path, decode_times=decode_times)
                var_data = self.process_variable(var_name, data_model, scaling_factor)
                
                if var_data is not None:
                    data_list.append(var_data)
                    model_ids.append(self.get_model_id(data_model))
                    lats.append(data_model.variables['lat'][:])
                    lons.append(data_model.variables['lon'][:])
                
                data_model.close()
                    
            except Exception as e:
                print(f"Error with file {path}: {str(e)}")
                continue

        return np.array(data_list), model_ids, lats, lons

def calculate_atmospheric_variables(climate_data):
    """Calculate atmospheric variables"""
    print("\n=== Calculating atmospheric variables ===")
    
    n = climate_data['metadata']['n']
    tos_change_array = climate_data['results']['tos']['change_array']
    tos_baseline = climate_data['results']['tos']['baseline']
    tos_Kelvin_array = tos_baseline + 273
    
    # Constants as in original
    sigma = 5.67*10**-8
    alpha = 0.06
    
    print("Calculating coefficients...")
    # LW up
    coef_LW_up_array = 4*sigma*(tos_Kelvin_array**3)
    coef_LW_up_array = np.where(np.isnan(tos_change_array), np.nan, coef_LW_up_array)
    
    # Upward LW atmospheric flux
    rlus_change_array = climate_data['results']['rlus']['change_array']
    atm_LW_up_array = rlus_change_array - coef_LW_up_array * tos_change_array
    
    # Latent
    print("Calculating latent...")
    hfls_baseline = climate_data['results']['hfls']['baseline']
    oceanic_latent_list = []
    oceanic_latent_coef_list = []
    
    for i in range(n):
        oceanic_latent = alpha * hfls_baseline[i] * tos_change_array[i]
        oceanic_latent_list.append(oceanic_latent)
        
        oceanic_latent_coef = -alpha * hfls_baseline[i]
        oceanic_latent_coef_list.append(oceanic_latent_coef)
    
    oceanic_latent_array = np.array(oceanic_latent_list)
    oceanic_latent_coef_array = np.array(oceanic_latent_coef_list)
    
    # Atmospheric latent heat
    hfls_change_array = climate_data['results']['hfls']['change_array']
    atm_latent_array = hfls_change_array - oceanic_latent_array
    
    # Total coefficients
    coef_up_array = oceanic_latent_coef_array + coef_LW_up_array
    
    print("Calculating D0...")
    # D0 as in original script
    D0_change_array = (
        -climate_data['results']['rsntds']['change_array'] +
        rlus_change_array -
        climate_data['results']['rlds']['change_array'] -
        hfls_change_array -
        climate_data['results']['hfss']['change_array']
    )
    
# Zonal means
    D0_mean_lat = np.nanmean(D0_change_array[:,55:104,0:360], axis=1)
    D0_mean_state = np.nanmean(D0_mean_lat, axis=1)
    
    atm_LW_up_mean_lat = np.nanmean(atm_LW_up_array[:,55:104,0:360], axis=1)
    atm_LW_up_mean_state = np.nanmean(atm_LW_up_mean_lat, axis=1)
    
    # Atmospheric variables exactly as in original script
    atm_variable_change = np.array([
        climate_data['results']['rsntds']['change_array'],
        climate_data['results']['hfss']['change_array'],
        climate_data['results']['rlds']['change_array'],
        atm_latent_array,
        atm_LW_up_array
    ])
    
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
    """Calculate patterns exactly as in original script"""
    print("\n=== Calculating patterns and deviations ===")
    
    n = climate_data['metadata']['n']
    tos_change_array = climate_data['results']['tos']['change_array']
    
    # Land filtering as in original
    print("Filtering land for TOS...")
    tos_change_array_fitre_land = np.where(tos_change_array == 0, np.nan, tos_change_array)
    
    # Mean states as in original
    print("Calculating mean states...")
    tos_change_mean_state_lat = np.nanmean(tos_change_array_fitre_land[:,55:104,0:360], axis=1)
    tos_change_mean_state = np.nanmean(tos_change_mean_state_lat, axis=1)
    tos_change_mean_state_mean = np.nanmean(tos_change_mean_state)
    
    # TOS relative patterns as in original
    relative_tos_change_list = []
    for i in range(n):
        relative_tos_change = tos_change_array[i] - tos_change_mean_state[i]
        relative_tos_change_list.append(relative_tos_change)
    relative_tos_change_array = np.array(relative_tos_change_list)
    
    # Relative atmospheric variables as in original
    atm_variable_change = atm_results['atmospheric']['variables']
    atm_variable_change_mean_state_lat = np.nanmean(atm_variable_change[:,:,55:104,0:360], axis=2)
    atm_variable_change_mean_state = np.nanmean(atm_variable_change_mean_state_lat, axis=2)
    
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
    
    # Atmospheric patterns as in original
    total_coef_array = atm_results['atmospheric']['coef_up']
    total_coef_array_mean_state_lat = np.nanmean(total_coef_array[:,55:104,0:360], axis=1)
    total_coef_array_mean_state = np.nanmean(total_coef_array_mean_state_lat, axis=1)
    total_coef_array_MMM = np.nanmean(total_coef_array, axis=0)
    
    # Relative coefficients as in original
    relative_total_coef_list = []
    for i in range(n):
        relative_total_coef = total_coef_array[i] - total_coef_array_mean_state[i]
        relative_total_coef_list.append(relative_total_coef)
    relative_total_coef_array = np.array(relative_total_coef_list)
    
    # Final patterns as in original
    Patterns_atm_variable_total_list = []
    for j in range(5):
        relative_var = relative_atm_variable_array[j]
        Patterns_atm_variable_list = []
        for i in range(n):
            Patterns_atm_variable = relative_var[i]/total_coef_array_MMM
            Patterns_atm_variable_list.append(Patterns_atm_variable)
        Patterns_atm_variable_total_list.append(Patterns_atm_variable_list)
    Patterns_atm_variable_total_array = np.array(Patterns_atm_variable_total_list)
    
    # Spatial heterogeneity as in original
    print("Calculating spatial heterogeneity...")
    Spatial_heterogeneity_array = relative_total_coef_array * relative_tos_change_array
    Spatial_heterogeneity_array_mean_state_lat = np.nanmean(Spatial_heterogeneity_array[:,55:104,0:360], axis=1)
    Spatial_heterogeneity_array_mean_state = np.nanmean(Spatial_heterogeneity_array_mean_state_lat, axis=1)
    
    
# RSST Zhang calculation as in original
    print("Calculating RSST Zhang...")
    tos_Zhang_change_array = ((climate_data['results']['rsntds']['change_array'] +
                              climate_data['results']['rlds']['change_array'] +
                              atm_results['atmospheric']['latent']['atmospheric'] +
                              climate_data['results']['hfss']['change_array'] +
                              atm_results['atmospheric']['D0']['change'] -
                              atm_results['atmospheric']['radiation']['LW_up']['atmospheric']) /
                             total_coef_array)
    
    tos_Zhang_change_mean_state_lat = np.nanmean(tos_Zhang_change_array[:,55:104,0:360], axis=1)
    tos_Zhang_change_mean_state = np.nanmean(tos_Zhang_change_mean_state_lat, axis=1)
    
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
    """Calculate wind variables exactly as in original script"""
    print("\n=== Calculating wind variables ===")
    
    # Number of models for standard variables
    n = climate_data['metadata']['n']
    # Number of models for uas/vas (different because fewer models available)
    n_wind = len(climate_data['results']['vas']['change_array'])
    
    print(f"Number of models for tau: {n}")
    print(f"Number of models for surface wind: {n_wind}")
    
    # Calculations for tauu as in original (uses n)
    print("Calculating wind stress (tau)...")
    tauu_change_array = climate_data['results']['tauu']['change_array']
    tauu_change_mean_state_lat = np.nanmean(tauu_change_array[:,55:104,0:360], axis=1)
    tauu_change_mean_state = np.nanmean(tauu_change_mean_state_lat, axis=1)
    
    # Relative changes tauu
    relative_tauu_change_list = []
    for i in range(n):  # uses n
        relative_tauu_change = tauu_change_array[i] - tauu_change_mean_state[i]
        relative_tauu_change_list.append(relative_tauu_change)
    relative_tauu_change_array = np.array(relative_tauu_change_list)
    
    # Calculations for tauv
    tauv_change_array = climate_data['results']['tauv']['change_array']
    tauv_change_mean_state_lat = np.nanmean(tauv_change_array[:,55:104,0:360], axis=1)
    tauv_change_mean_state = np.nanmean(tauv_change_mean_state_lat, axis=1)
    
    relative_tauv_change_list = []
    for i in range(n):  # uses n
        relative_tauv_change = tauv_change_array[i] - tauv_change_mean_state[i]
        relative_tauv_change_list.append(relative_tauv_change)
    relative_tauv_change_array = np.array(relative_tauv_change_list)
    
    # Calculate total wind stress as in original
    print("Calculating total wind stress...")
    wind_stress_change_list = []
    for i in range(n):  # uses n
        wind_stress = np.sqrt(tauu_change_array[i]**2 + tauv_change_array[i]**2)
        wind_stress_change_list.append(wind_stress)
    wind_stress_change_array = np.array(wind_stress_change_list)
    
    # Normalization as in original
    print("Normalizing wind components...")
    tauu_change_array_norm = tauu_change_array / np.sqrt(tauu_change_array**2 + tauv_change_array**2)
    tauv_change_array_norm = tauv_change_array / np.sqrt(tauu_change_array**2 + tauv_change_array**2)
    
    
# Calculate surface wind as in original - uses n_wind
    print("Calculating surface wind variables...")
    uas_baseline = climate_data['results']['uas']['baseline']
    uas_future = climate_data['results']['uas']['future']
    uas_change_array = uas_future - uas_baseline
    
    uas_change_mean_state_lat = np.nanmean(uas_change_array[:,55:104,0:360], axis=1)
    uas_change_mean_state = np.nanmean(uas_change_mean_state_lat, axis=1)
    
    # Use n_wind for uas
    relative_uas_change_list = []
    for i in range(n_wind):  # uses n_wind
        relative_uas = uas_change_array[i] - uas_change_mean_state[i]
        relative_uas_change_list.append(relative_uas)
    relative_uas_change_array = np.array(relative_uas_change_list)
    
    # Same for VAS with n_wind
    vas_baseline = climate_data['results']['vas']['baseline']
    vas_future = climate_data['results']['vas']['future']
    vas_change_array = vas_future - vas_baseline
    
    vas_change_mean_state_lat = np.nanmean(vas_change_array[:,55:104,0:360], axis=1)
    vas_change_mean_state = np.nanmean(vas_change_mean_state_lat, axis=1)
    
    relative_vas_change_list = []
    for i in range(n_wind):  # uses n_wind
        relative_vas = vas_change_array[i] - vas_change_mean_state[i]
        relative_vas_change_list.append(relative_vas)
    relative_vas_change_array = np.array(relative_vas_change_list)
    
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
            'n_wind': n_wind
        }
    }

def calculate_radiation_variables(climate_data, total_coef_array_MMM):
    """Calculate radiation variables exactly as in original script"""
    print("\n=== Calculating radiation variables ===")
    
    n = climate_data['metadata']['n']
    rlus_change_array = climate_data['results']['rlus']['change_array']
    
    # RLUS zonal means as in original
    rlus_change_mean_state_lat = np.nanmean(rlus_change_array[:,55:104,0:360], axis=1)
    rlus_change_mean_state = np.nanmean(rlus_change_mean_state_lat, axis=1)
    
    # RLUS relative changes
    relative_rlus_change_list = []
    for i in range(n):
        relative_rlus = rlus_change_array[i] - rlus_change_mean_state[i]
        relative_rlus_change_list.append(relative_rlus)
    relative_rlus_change_array = np.array(relative_rlus_change_list)
    
    # RLUS patterns as in original
    Patterns_relative_rlus_change_list = []
    for i in range(n):
        pattern = relative_rlus_change_array[i]/total_coef_array_MMM
        Patterns_relative_rlus_change_list.append(pattern)
    Patterns_relative_rlus_change_array = np.array(Patterns_relative_rlus_change_list)
    
    

def load_cobe_sst_data(file_path):
    """
    Load COBE SST observation data and calculate time mean
    
    Parameters:
    file_path (str): Path to the COBE SST NetCDF file
    
    Returns:
    tuple: (tos_COBE_array, timmean_tos_COBE)
    """
    print("\n=== Chargement des données COBE SST ===")
    try:
        data_COBE = xr.open_dataset(file_path)
        
        # Extract SST data
        tos_COBE = data_COBE.variables['sst'][:]
        tos_COBE_array = np.array(tos_COBE)
        
        # Calculate time mean for 1900-1925
        timmean_tos_COBE_1900_1925 = np.nanmean(tos_COBE, axis=0)
        
        print("Données COBE SST chargées avec succès")
        return tos_COBE_array, timmean_tos_COBE_1900_1925
    
    except Exception as e:
        print(f"Erreur lors du chargement des données COBE: {str(e)}")
        raise
    
def main():
    """
    Main function to load and calculate all climate variables
    """
    print(f"\n=== PROCESSING START - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    try:
        # 1. Initial data loading
        print("\nStep 1: Loading base climate data...")
        processor = ClimateDataProcessor()
        
        # Loading TOS
        print("\n=== Loading TOS ===")
        results_tos_1925 = processor.process_period_data('tos', "1925")
        results_tos_2075 = processor.process_period_data('tos', "2075")
        
        climate_data = {
            'metadata': {
                'n': len(results_tos_1925[0]),
                'lat': results_tos_1925[2][0],
                'lon': results_tos_1925[3][0],
                'model_ids': results_tos_1925[1]
            },
            'results': {
                'tos': {
                    'change_array': results_tos_2075[0] - results_tos_1925[0],
                    'baseline': results_tos_1925[0],
                    'future': results_tos_2075[0]
                }
            }
        }
        
        # CMIP5/CMIP6 Classification
        print("\n=== CMIP5/CMIP6 Classification ===")
        paths_2075 = processor.get_file_paths('tos', "2075")
        icmip5 = []
        icmip6 = []
        
        for i, path in enumerate(paths_2075):
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
        
        # Loading other variables
        print("\n=== Loading additional variables ===")
        variables = ['hfls', 'hfss', 'rlds', 'rlus', 'rsntds', 'tauu', 'tauv', 'uas', 'vas','clt',"pr"]
        
        for var in variables:
            print(f"Processing {var}...")
            results_1925 = processor.process_period_data(var, "1925")
            results_2075 = processor.process_period_data(var, "2075")
            
            climate_data['results'][var] = {
                'change_array': results_2075[0] - results_1925[0],
                'baseline': results_1925[0],
                'future': results_2075[0]
            }
        
        # 2. Atmospheric calculations
        print("\nStep 2: Atmospheric calculations...")
        atm_results = calculate_atmospheric_variables(climate_data)
        
        # 3. Pattern calculations
        print("\nStep 3: Pattern calculations...")
        patterns_results = calculate_patterns_and_deviations(climate_data, atm_results)
        
        # 4. Wind variables
        print("\nStep 4: Wind calculations...")
        wind_results = calculate_wind_variables(climate_data)
        
        # 5. Radiation variables
        print("\nStep 5: Radiation calculations...")
        radiation_results = calculate_radiation_variables(
            climate_data, 
            patterns_results['coefficients']['MMM']
        )
        
        # Assembling final results
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
        
        # Load COBE SST data
         cobe_file_path = "file_1900_1925.nc"
        # cobe_file_path = "file_1960_1980.nc"

        
        tos_COBE_array, timmean_tos_COBE_1900_1925 = load_cobe_sst_data(cobe_file_path)
        
        # Calculate TOS bias
        tos_biais_array = results_tos_1925[0] - timmean_tos_COBE_1900_1925
        tos_biais_array_MMM = np.nanmean(tos_biais_array, axis=0)
        
        # Add bias results to the climate_data dictionary
        climate_data['results']['tos'].update({
            'bias_array': tos_biais_array,
            'bias_MMM': tos_biais_array_MMM,
            'COBE_baseline': timmean_tos_COBE_1900_1925
        })
        
        print(f"\n=== PROCESSING COMPLETE - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        return results
        
    except Exception as e:
        print(f"\n!!! ERROR DURING PROCESSING: {str(e)}")
        raise

if __name__ == "__main__":
    results = main()