#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 12:20:07 2025

@author: vincent
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 20:43:49 2025

@author: vincent
"""

import numpy as np
import importlib.util
import xarray as xr

def load_and_process_data():
    """Loads and makes all calculated variables globally accessible"""
    
    # Global variables for storing file data including:
    # - Climate variables (temperature, radiation, heat fluxes)
    # - Wind stress components
    # - Cloud coverage
    # - Model metadata
    global n, n_wind, results  # n = number of models, n_wind = number of wind models
    global rsntds_change_array, rlus_change_array, rlds_change_array
    global hfls_change_array, hfss_change_array
    global tos_change_array, tos_Kelvin_array, hfls_1900_1925_array
    global tauu_change_array, tauv_change_array
    global uas_change_array, vas_change_array
    global uas_1900_1925_array, vas_1900_1925_array
    global uas_2075_2100_array, vas_2075_2100_array
    global clt_change_array
    global tos_COBE_change_array
    global relative_tos_COBE_change_array


    global lons_tos_2075_2100, lats_tos_2075_2100
    global icmip5, icmip6
    
    # # Load initial data from external module
    # spec = importlib.util.spec_from_file_location(
    #     "load_data_module",
    #     "/load_data_histo.py"
    # )
    
    spec = importlib.util.spec_from_file_location(
        "load_data_module",
        "/load_data_histo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    results = module.main()
    
    # Extract model metadata
    n = len(results['climate_data']['metadata']['model_ids'])
    n_wind = len(results['wind']['surface']['u']['change'])
    icmip5 = results['climate_data']['metadata']['icmip5']
    icmip6 = results['climate_data']['metadata']['icmip6']
    
    model_list = results['climate_data']['metadata']['model_ids']
    
    
    # Extract climate variables
    # tos = Sea Surface Temperature
    # rsntds = Net Shortwave Radiation
    # rlus/rlds = Upward/Downward Longwave Radiation
    # hfls/hfss = Latent/Sensible Heat Flux
    tos_change_array = results['climate_data']['results']['tos']['change_array']
    tos_Kelvin_array = results['climate_data']['results']['tos']['baseline'] + 273.15
    rsntds_change_array = results['climate_data']['results']['rsntds']['change_array']
    rlus_change_array = results['climate_data']['results']['rlus']['change_array']
    rlds_change_array = results['climate_data']['results']['rlds']['change_array']
    hfls_change_array = results['climate_data']['results']['hfls']['change_array']
    hfss_change_array = results['climate_data']['results']['hfss']['change_array']
    
    hfls_1900_1925_array = results['climate_data']['results']['hfls']['baseline']
    
    tauu_change_array = results['climate_data']['results']['tauu']['change_array']
    tauv_change_array = results['climate_data']['results']['tauv']['change_array']
    
    uas_change_array = results['climate_data']['results']['uas']['change_array']
    vas_change_array = results['climate_data']['results']['vas']['change_array']
    
    uas_1900_1925_array = results['climate_data']['results']['uas']['baseline']
    vas_1900_1925_array = results['climate_data']['results']['vas']['baseline']
    
    uas_2075_2100_array = results['climate_data']['results']['uas']['future']
    vas_2075_2100_array = results['climate_data']['results']['vas']['future']
    
    clt_change_array = results['climate_data']['results']['clt']['change_array']
 
    
    # Coordonnées
    lats_tos_2075_2100 = results['climate_data']['metadata']['lat']
    lons_tos_2075_2100 = results['climate_data']['metadata']['lon']
    
    print("Chargement et déclaration des variables terminés")
    return results, model_list  # Return both results and model_list


def init():
    """Declares all global variables used in climate analysis"""
    
    global model_list

    # D0 variables (net surface heat flux)
    global D0_change_array, D0_change_array_MMM  # MMM = Multi-Model Mean
    global D0_change_array_EQUATOR, D0_change_EQUATOR_zonal
    global D0_change_EQUATOR_zonal_model_mean, D0_change_EQUATOR_zonal_model_STD
    
    # Total coefficient variables
    global total_coef_array, total_coef_array_MMM, total_coef_array_EQUATOR
    global total_coef_array_mean_state_lat_PACIFIC, total_coef_array_PACIFIC_mean_state_MMM
    global total_coef_array_mean_state_lat, total_coef_array_mean_state
    global total_coef_array_mean_state_MMM

    # Atmospheric variables
    global atm_variable_change, atm_variable_change_MMM
    global atm_variable_array_EQUATOR, atm_variable_EQUATOR_zonal
    global atm_variable_EQUATOR_zonal_model_mean, atm_variable_EQUATOR_zonal_model_STD

    # Zhang variables (temperature decomposition)
    global relative_total_coef_array, relative_total_coef_array_MMM, relative_tos_change_array_STD
    global tos_Zhang_change_array, tos_Zhang_change_array_MMM
    global tos_Zhang_EQUATOR_change, tos_Zhang_EQUATOR_change_zonal
    global tos_Zhang_EQUATOR_change_zonal_model_mean, tos_Zhang_EQUATOR_change_zonal_model_STD

    # Fraction variables (normalized quantities)
    global atm_variable_change_fraction, atm_variable_change_fraction_MMM
    global atm_variable_array_EQUATOR_fraction, atm_variable_array_EQUATOR_fraction_zonal
    global atm_variable_array_EQUATOR_fraction_model_mean
    global atm_variable_array_EQUATOR_fraction_zonal_model_STD
    global atm_fraction_name

    # D0 fraction variables
    global D0_change_array_fraction, D0_change_array_fraction_MMM
    global D0_change_array_EQUATOR_fraction, D0_change_array_EQUATOR_fraction_zonal
    global D0_change_array_EQUATOR_fraction_model_mean
    global D0_change_array_EQUATOR_fraction_zonal_model_STD

    # Mean state variables
    global D0_change_array_mean_state_lat, D0_change_array_mean_state
    global D0_change_array_fraction_mean_state, D0_change_array_fraction_mean_state_MMM
    global D0_change_array_mean_state_lat_PACIFIC, D0_change_array_PACIFIC_mean_state_MMM
    global D0_change_array_fraction_PACIFIC_mean_state_MMM
    global D0_change_array_fraction_PACIFIC_mean_state_MMM_MMM

    # Atmospheric mean state variables
    global atm_variable_change_mean_state_lat, atm_variable_change_mean_state
    global atm_variable_change_fraction_mean_state
    global atm_variable_change_mean_state_lat_PACIFIC, atm_variable_change_PACIFIC_mean_state_MMM
    global atm_variable_change_fraction_PACIFIC_mean_state_MMM
    
    # Pattern sum variables
    global Pattern_atm_sum_component_change, Pattern_atm_sum_component_change_MMM
    global oceanic_sum_component_change, oceanic_sum_component_change_MMM
    
    # Ocean deviation variables
    global oceanic_flux_deviation_array, oceanic_flux_deviation_array_fraction
    global oceanic_flux_deviation_array_fraction_MMM, oceanic_flux_deviation_array_fraction_STD
   
    # Fractional spatial heterogeneity variables
    global Spatial_heterogeneity_fraction_array, Spatial_heterogeneity_fraction_array_MMM
    global Spatial_heterogeneity_fraction_array_MMM_CMIP5, Spatial_heterogeneity_fraction_array_MMM_CMIP6
   
    # RSST (Relative Sea Surface Temperature) variables
    global RSST_sum_component_change_ALPHA_MMM
    global RSST_sum_component_change_ALPHA_MMM_MMM, RSST_sum_component_change_ALPHA_MMM_STD
    
    # Relative atmospheric variables
    global relative_atm_variable_array, relative_atm_variable_array_MMM
    global Patterns_atm_variable_total_array, Patterns_atm_variable_total_array_MMM
    global Patterns_atm_variable_total_array_MMM_CMIP5, Patterns_atm_variable_total_array_MMM_CMIP6
    
    # SST bias variables
    global tos_COBE_change_array
    global relative_tos_COBE_change_array
    
    global clt_change_array  # Cloud cover change array
       
    

def calculate_D0_changes():
    """Calculates D0 changes (net surface heat flux)
    D0 = -rsntds + rlus - rlds - hfls - hfss"""
    global D0_change_array, D0_change_array_MMM
    
    D0_change_array = (-rsntds_change_array + rlus_change_array - rlds_change_array - 
                      hfls_change_array - hfss_change_array)
    D0_change_array_MMM = np.nanmean(D0_change_array, axis=0)  # Multi-Model Mean
    
        
    # Région équatoriale
    global D0_change_array_EQUATEUR, D0_change_EQUATEUR_zonal
    global D0_change_EQUATEUR_zonal_model_mean, D0_change_EQUATEUR_zonal_model_STD
    
    D0_change_array_EQUATEUR = D0_change_array[:,75:85,120:280]
    D0_change_EQUATEUR_zonal = np.nanmean(D0_change_array_EQUATEUR, axis=1)
    D0_change_EQUATEUR_zonal_model_mean = np.nanmean(D0_change_EQUATEUR_zonal, axis=0)
    D0_change_EQUATEUR_zonal_model_STD = np.std(D0_change_EQUATEUR_zonal, axis=0)

def calculate_atmospheric_variables():
    """Processes atmospheric variables including radiation and heat fluxes"""
    global atm_variable_change, atm_variable_change_MMM
    
    atm_variable_change = np.array([rsntds_change_array, hfss_change_array,
                                   rlds_change_array, atm_latent_array, atm_LW_up_array])
    atm_variable_change_MMM = np.nanmean(atm_variable_change, axis=1)
    
    # Région équatoriale
    global atm_variable_array_EQUATEUR, atm_variable_EQUATEUR_zonal
    global atm_variable_EQUATEUR_zonal_model_mean, atm_variable_EQUATEUR_zonal_model_STD
    
    atm_variable_array_EQUATEUR = atm_variable_change[:,:,75:85,120:280]
    atm_variable_EQUATEUR_zonal = np.nanmean(atm_variable_array_EQUATEUR, axis=2)
    atm_variable_EQUATEUR_zonal_model_mean = np.nanmean(atm_variable_EQUATEUR_zonal, axis=1)
    atm_variable_EQUATEUR_zonal_model_STD = np.std(atm_variable_EQUATEUR_zonal, axis=1)

    
    
def calculate_total_coefficients():
    """Calculates total coefficients"""
    global total_coef_array, total_coef_array_MMM
    
    total_coef_array = coef_LW_up_array + oceanic_latent_coef_array
    total_coef_array_MMM = np.nanmean(total_coef_array, axis=0)
    
    # Moyennes spatiales
    global total_coef_array_mean_state_lat_PACIFIC, total_coef_array_PACIFIC_mean_state_MMM
    global total_coef_array_mean_state_lat, total_coef_array_mean_state
    global total_coef_array_mean_state_MMM
    
    total_coef_array_mean_state_lat_PACIFIC = np.nanmean(total_coef_array[:,55:104,160:270], axis=1)
    total_coef_array_PACIFIC_mean_state_MMM = np.nanmean(total_coef_array_mean_state_lat_PACIFIC, axis=1)
    
    total_coef_array_mean_state_lat = np.nanmean(total_coef_array[:,55:104,0:360], axis=1)
    total_coef_array_mean_state = np.nanmean(total_coef_array_mean_state_lat, axis=1)
    total_coef_array_mean_state_MMM = np.nanmean(total_coef_array_mean_state)
    
def calculate_zhang_variables():
    """Calcule les variables de Zhang"""
    global tos_Zhang_change_array, tos_Zhang_change_array_MMM
    
    tos_Zhang_change_array = ((rsntds_change_array + rlds_change_array + atm_latent_array +
                              hfss_change_array + D0_change_array - atm_LW_up_array) /
                             total_coef_array)
    tos_Zhang_change_array_MMM = np.nanmean(tos_Zhang_change_array, axis=0)
    
    # Région équatoriale
    global tos_Zhang_EQUATEUR_change, tos_Zhang_EQUATEUR_change_zonal
    global tos_Zhang_EQUATEUR_change_zonal_model_mean, tos_Zhang_EQUATEUR_change_zonal_model_STD
    
    tos_Zhang_EQUATEUR_change = tos_Zhang_change_array[:,75:85,120:280]
    tos_Zhang_EQUATEUR_change_zonal = np.nanmean(tos_Zhang_EQUATEUR_change, axis=1)
    tos_Zhang_EQUATEUR_change_zonal_model_mean = np.nanmean(tos_Zhang_EQUATEUR_change_zonal, axis=0)
    tos_Zhang_EQUATEUR_change_zonal_model_STD = np.std(tos_Zhang_EQUATEUR_change_zonal, axis=0)

def calculate_pattern_sum():
    """Calcule les sommes des composantes des patterns"""
    global Pattern_atm_somme_component_change, Pattern_atm_somme_component_change_MMM
    global oceanic_somme_component_change, oceanic_somme_component_change_MMM
    
    # Calcul des sommes atmosphériques
    Pattern_atm_somme_component_change_list = []
    for i in range(n):
        test = (
            Patterns_atm_variable_total_array[0,i] +
            Patterns_atm_variable_total_array[1,i] +
            Patterns_atm_variable_total_array[2,i] +
            Patterns_atm_variable_total_array[3,i] -
            Patterns_atm_variable_total_array[4,i]
        )
        Pattern_atm_somme_component_change_list.append(test)
    Pattern_atm_somme_component_change = np.array(Pattern_atm_somme_component_change_list)
    Pattern_atm_somme_component_change_MMM = np.nanmean(Pattern_atm_somme_component_change, axis=0)

    # Calcul des sommes océaniques
    oceanic_somme_component_change_list = []
    for i in range(n):
        test = (
            Spatial_heterogeneity_fraction_array[i] +
            oceanic_flux_deviation_array_fraction[i]
        )
        oceanic_somme_component_change_list.append(test)
    oceanic_somme_component_change = np.array(oceanic_somme_component_change_list)
    oceanic_somme_component_change_MMM = np.nanmean(oceanic_somme_component_change, axis=0)
    
def calculate_total_coefficients():
    """Calcule les coefficients totaux"""
    global total_coef_array, total_coef_array_MMM
    global total_coef_array_EQUATEUR  # Ajout de cette variable globale
    
    total_coef_array = coef_LW_up_array + oceanic_latent_coef_array
    total_coef_array_MMM = np.nanmean(total_coef_array, axis=0)
    
    # Région équatoriale
    total_coef_array_EQUATEUR = total_coef_array[:,75:85,120:280]
    
    # Moyennes spatiales
    global total_coef_array_mean_state_lat_PACIFIC, total_coef_array_PACIFIC_mean_state_MMM
    global total_coef_array_mean_state_lat, total_coef_array_mean_state
    global total_coef_array_mean_state_MMM
    
    total_coef_array_mean_state_lat_PACIFIC = np.nanmean(total_coef_array[:,55:104,160:270], axis=1)
    total_coef_array_PACIFIC_mean_state_MMM = np.nanmean(total_coef_array_mean_state_lat_PACIFIC, axis=1)
    
    total_coef_array_mean_state_lat = np.nanmean(total_coef_array[:,55:104,0:360], axis=1)
    total_coef_array_mean_state = np.nanmean(total_coef_array_mean_state_lat, axis=1)
    total_coef_array_mean_state_MMM = np.nanmean(total_coef_array_mean_state)
    
    
def calculate_fractions():
    """Calcule les fractions des variables"""
    global atm_variable_change_fraction, atm_variable_change_fraction_MMM
    global atm_variable_array_EQUATEUR_fraction, atm_variable_array_EQUATEUR_fraction_zonal
    global atm_variable_array_EQUATEUR_fraction_model_mean
    global atm_variable_array_EQUATEUR_fraction_zonal_model_STD
    
    atm_variable_change_fraction = atm_variable_change/total_coef_array_MMM
    atm_variable_change_fraction_MMM = np.nanmean(atm_variable_change_fraction, axis=1)

    atm_variable_array_EQUATEUR_fraction = atm_variable_array_EQUATEUR/total_coef_array_EQUATEUR
    atm_variable_array_EQUATEUR_fraction_zonal = np.nanmean(atm_variable_array_EQUATEUR_fraction, axis=2)
    atm_variable_array_EQUATEUR_fraction_model_mean = np.nanmean(atm_variable_array_EQUATEUR_fraction_zonal, axis=1)
    atm_variable_array_EQUATEUR_fraction_zonal_model_STD = np.std(atm_variable_array_EQUATEUR_fraction_zonal, axis=1)

    # D0 fractions
    global D0_change_array_fraction, D0_change_array_fraction_MMM
    global D0_change_array_EQUATEUR_fraction, D0_change_array_EQUATEUR_fraction_zonal
    global D0_change_array_EQUATEUR_fraction_model_mean, D0_change_array_EQUATEUR_fraction_zonal_model_STD
    
    D0_change_array_fraction = D0_change_array/total_coef_array_MMM
    D0_change_array_fraction_MMM = np.nanmean(D0_change_array_fraction, axis=0)
    D0_change_array_EQUATEUR_fraction = D0_change_array_EQUATEUR/total_coef_array_EQUATEUR
    D0_change_array_EQUATEUR_fraction_zonal = np.nanmean(D0_change_array_EQUATEUR_fraction, axis=1)
    D0_change_array_EQUATEUR_fraction_model_mean = np.nanmean(D0_change_array_EQUATEUR_fraction_zonal, axis=0)
    D0_change_array_EQUATEUR_fraction_zonal_model_STD = np.std(D0_change_array_EQUATEUR_fraction_zonal, axis=0)

def calculate_mean_state_patterns():
    """Calcule les patterns de l'état moyen"""
    global D0_change_array_mean_state_lat, D0_change_array_mean_state
    global D0_change_array_fraction_mean_state, D0_change_array_fraction_mean_state_MMM
    
    D0_change_array_mean_state_lat = np.nanmean(D0_change_array[:,55:104,0:360], axis=1)
    D0_change_array_mean_state = np.nanmean(D0_change_array_mean_state_lat, axis=1)

    D0_change_array_fraction_mean_state = D0_change_array_mean_state/total_coef_array_mean_state_MMM
    D0_change_array_fraction_mean_state_MMM = np.nanmean(D0_change_array_fraction_mean_state, axis=0)

    global D0_change_array_mean_state_lat_PACIFIC, D0_change_array_PACIFIC_mean_state_MMM
    global D0_change_array_fraction_PACIFIC_mean_state_MMM
    global D0_change_array_fraction_PACIFIC_mean_state_MMM_MMM
    
    D0_change_array_mean_state_lat_PACIFIC = np.nanmean(D0_change_array[:,55:104,160:270], axis=1)
    D0_change_array_PACIFIC_mean_state_MMM = np.nanmean(D0_change_array_mean_state_lat_PACIFIC, axis=1)


def calculate_wind_components():
    """Calculates wind stress magnitude from u and v components"""
    global wind_stress_change_array, wind_stress_change_array_mean
    
    wind_stress_change_list = []
    for i in range(n):
        magnitude = np.sqrt(tauu_change_array[i]**2 + tauv_change_array[i]**2)
        wind_stress_change_list.append(magnitude)
    wind_stress_change_array = np.array(wind_stress_change_list)
    wind_stress_change_array_mean = np.nanmean(wind_stress_change_array, axis=0)
    
def calculate_normalized_wind():
    """Calcule les vents normalisés"""
    global tauu_change_array_norm, tauu_change_array_norm_median
    global tauv_change_array_norm, tauv_change_array_norm_median
    
    tauu_change_array_norm = tauu_change_array/np.sqrt(tauu_change_array**2 + tauv_change_array**2)
    tauu_change_array_norm_median = np.nanmean(tauu_change_array_norm, axis=0)
    tauv_change_array_norm = tauv_change_array/np.sqrt(tauu_change_array**2 + tauv_change_array**2)
    tauv_change_array_norm_median = np.nanmean(tauv_change_array_norm, axis=0)
    
def calculate_wind_stress():
    """Calcule les contraintes de vent"""
    global tauu_change_mean_state_lat, tauu_change_mean_state, tauu_change_mean_state_MMM
    global tauu_change_array_tropic, relative_tauu_change_array, relative_tauu_change_array_MMM
    
    # Calculs pour tauu
    tauu_change_mean_state_lat = np.nanmean(tauu_change_array[:,55:104,0:360], axis=1)
    tauu_change_mean_state = np.nanmean(tauu_change_mean_state_lat, axis=1)
    tauu_change_mean_state_MMM = np.nanmean(tauu_change_mean_state, axis=0)
    tauu_change_array_tropic = tauu_change_array[:,55:104,0:360]

    relative_tauu_change_list = []
    for i in range(n):
        relative_tauu_change = tauu_change_array[i] - tauu_change_mean_state[i]
        relative_tauu_change_list.append(relative_tauu_change)
    relative_tauu_change_array = np.array(relative_tauu_change_list)
    relative_tauu_change_array_MMM = np.nanmean(relative_tauu_change_array, axis=0)

    # Calculs pour tauv
    global tauv_change_mean_state_lat, tauv_change_mean_state, tauv_change_mean_state_MMM
    global tauv_change_array_tropic, relative_tauv_change_array, relative_tauv_change_array_MMM
    
    tauv_change_mean_state_lat = np.nanmean(tauv_change_array[:,55:104,0:360], axis=1)
    tauv_change_mean_state = np.nanmean(tauv_change_mean_state_lat, axis=1)
    tauv_change_mean_state_MMM = np.nanmean(tauv_change_mean_state, axis=0)
    tauv_change_array_tropic = tauv_change_array[:,55:104,0:360]

def calculate_spatial_heterogeneity():
    """Calcule l'hétérogénéité spatiale"""
    global Spatial_heterogeneity_array
    global Spatial_heterogeneity_array_mean_state_lat, Spatial_heterogeneity_array_mean_state
    
    Spatial_heterogeneity_array = relative_total_coef_array * relative_tos_change_array
    
    Spatial_heterogeneity_array_mean_state_lat = np.nanmean(Spatial_heterogeneity_array[:,55:104,0:360], axis=1)
    Spatial_heterogeneity_array_mean_state = np.nanmean(Spatial_heterogeneity_array_mean_state_lat, axis=1)
    
def calculate_LW_coefficients():
    """Calcule les coefficients de radiation LW"""
    global sigma, coef_LW_up_array, coef_LW_up_array_rapport, coef_LW_up_array_MMM
    global coef_LW_up_array_EQUATEUR, coef_LW_up_array_EQUATEUR_zonal
    global coef_LW_up_array_EQUATEUR_zonal_model_mean, coef_LW_up_array_EQUATEUR_zonal_model_STD
    global atm_LW_up_array, atm_LW_up_array_MMM
    global atm_LW_up_array_mean_lat_Pacific, atm_LW_up_array_Pacific_mean
    global atm_LW_up_array_mean_lat, atm_LW_up_array_mean
    
    # Coefficients LW up
    sigma = 5.67e-8
    coef_LW_up_array = 4 * sigma * (tos_Kelvin_array**3)
    coef_LW_up_array_rapport = rlus_change_array/tos_change_array
    coef_LW_up_array_MMM = np.nanmean(coef_LW_up_array, axis=0)

    # Région équatoriale
    coef_LW_up_array_EQUATEUR = coef_LW_up_array[:,75:85,120:280]
    coef_LW_up_array_EQUATEUR_zonal = np.nanmean(coef_LW_up_array_EQUATEUR, axis=1)
    coef_LW_up_array_EQUATEUR_zonal_model_mean = np.nanmean(coef_LW_up_array_EQUATEUR_zonal, axis=0)
    coef_LW_up_array_EQUATEUR_zonal_model_STD = np.std(coef_LW_up_array_EQUATEUR_zonal, axis=0)

    # Calculs atmosphériques LW
    atm_LW_up_array = rlus_change_array - coef_LW_up_array * tos_change_array
    atm_LW_up_array_MMM = np.nanmean(atm_LW_up_array, axis=0)

    atm_LW_up_array_mean_lat_Pacific = np.nanmean(atm_LW_up_array[:,55:104,160:270], axis=1)
    atm_LW_up_array_Pacific_mean = np.nanmean(atm_LW_up_array_mean_lat_Pacific, axis=1)
    atm_LW_up_array_mean_lat = np.nanmean(atm_LW_up_array[:,55:104,0:360], axis=1)
    atm_LW_up_array_mean = np.nanmean(atm_LW_up_array_mean_lat, axis=1)

def calculate_latent_heat():
    """Calcule les coefficients de chaleur latente"""
    global alpha, oceanic_latent_array, oceanic_latent_array_multimodel_mean
    global oceanic_latent_coef_array, oceanic_latent_coef_array_MMM
    global oceanic_latent_coef_array_EQUATEUR, oceanic_latent_coef_array_EQUATEUR_zonal
    global oceanic_latent_coef_array_EQUATEUR_zonal_model_mean
    global oceanic_latent_coef_array_EQUATEUR_zonal_model_STD
    global atm_latent_array, coef_up_array
    
    # Chaleur latente océanique
    alpha = 0.06
    oceanic_latent_list = []
    oceanic_latent_coef_list = []
    for i in range(n):
        oceanic_latent = alpha * hfls_1900_1925_array[i] * tos_change_array[i]
        oceanic_latent_list.append(oceanic_latent)
        
        oceanic_latent_coef = -alpha * hfls_1900_1925_array[i]
        oceanic_latent_coef_list.append(oceanic_latent_coef)
    
    oceanic_latent_array = np.array(oceanic_latent_list)
    oceanic_latent_array_multimodel_mean = np.nanmean(oceanic_latent_array, axis=0)
    
    oceanic_latent_coef_array = np.array(oceanic_latent_coef_list)
    oceanic_latent_coef_array_MMM = np.nanmean(oceanic_latent_coef_array, axis=0)
    
    # Région équatoriale
    oceanic_latent_coef_array_EQUATEUR = oceanic_latent_coef_array[:,75:85,120:280]
    oceanic_latent_coef_array_EQUATEUR_zonal = np.nanmean(oceanic_latent_coef_array_EQUATEUR, axis=1)
    oceanic_latent_coef_array_EQUATEUR_zonal_model_mean = np.nanmean(oceanic_latent_coef_array_EQUATEUR_zonal, axis=0)
    oceanic_latent_coef_array_EQUATEUR_zonal_model_STD = np.std(oceanic_latent_coef_array_EQUATEUR_zonal, axis=0)

    # Chaleur latente atmosphérique
    atm_latent_array = hfls_change_array - oceanic_latent_array
    coef_up_array = oceanic_latent_coef_array + coef_LW_up_array
    
    
    
def calculate_additional_patterns():
    """Calcule les patterns de déviation océanique et les composantes RSST"""
    # Déviations océaniques
    global oceanic_flux_deviation_array, oceanic_flux_deviation_array_fraction
    global oceanic_flux_deviation_array_fraction_MMM, oceanic_flux_deviation_array_fraction_STD
    
    # Hétérogénéité spatiale
    global Spatial_heterogeneity_fraction_array, Spatial_heterogeneity_fraction_array_MMM
    global Spatial_heterogeneity_fraction_array_MMM_CMIP5, Spatial_heterogeneity_fraction_array_MMM_CMIP6
    
    # RSST
    global RSST_somme_component_change_ALPHA_MMM
    global RSST_somme_component_change_ALPHA_MMM_MMM, RSST_somme_component_change_ALPHA_MMM_STD
    
    # Calcul des déviations océaniques
    oceanic_flux_deviation_list = []
    for i in range(n):
        oceanic_flux_deviation = -tos_Zhang_change_mean_state[i] * relative_total_coef_array[i]
        oceanic_flux_deviation_list.append(oceanic_flux_deviation)
    oceanic_flux_deviation_array = np.array(oceanic_flux_deviation_list)
    
    # Fraction des déviations
    oceanic_flux_deviation_array_fraction_list = []
    for i in range(n):
        oceanic_flux_deviation_array_fraction_model = oceanic_flux_deviation_array[i] / total_coef_array_MMM
        oceanic_flux_deviation_array_fraction_list.append(oceanic_flux_deviation_array_fraction_model)
    oceanic_flux_deviation_array_fraction = np.array(oceanic_flux_deviation_array_fraction_list)
    
    oceanic_flux_deviation_array_fraction_MMM = np.nanmean(oceanic_flux_deviation_array_fraction, axis=0)
    oceanic_flux_deviation_array_fraction_STD = np.std(oceanic_flux_deviation_array_fraction, axis=0)
    
    # Calcul des fractions d'hétérogénéité spatiale
    Spatial_heterogeneity_fraction_list = []
    for i in range(n):
        test = Spatial_heterogeneity_array_mean_state[i] / total_coef_array_MMM
        Spatial_heterogeneity_fraction_list.append(test)
    Spatial_heterogeneity_fraction_array = np.array(Spatial_heterogeneity_fraction_list)
    
    Spatial_heterogeneity_fraction_array_MMM = np.nanmean(Spatial_heterogeneity_fraction_array, axis=0)
    Spatial_heterogeneity_fraction_array_MMM_CMIP5 = np.nanmean(Spatial_heterogeneity_fraction_array[icmip5], axis=0)
    Spatial_heterogeneity_fraction_array_MMM_CMIP6 = np.nanmean(Spatial_heterogeneity_fraction_array[icmip6], axis=0)
    
    # Calcul des composantes RSST
    RSST_somme_component_change_ALPHA_MMM_list = []
    for i in range(n):
        test = (Patterns_relative_D0_change_array[i] +
                Patterns_atm_variable_total_array[0,i] +
                Patterns_atm_variable_total_array[1,i] +
                Patterns_atm_variable_total_array[2,i] +
                Patterns_atm_variable_total_array[3,i] -
                Patterns_atm_variable_total_array[4,i] +
                Spatial_heterogeneity_fraction_array[i] +
                oceanic_flux_deviation_array_fraction[i])
        RSST_somme_component_change_ALPHA_MMM_list.append(test)
    RSST_somme_component_change_ALPHA_MMM = np.array(RSST_somme_component_change_ALPHA_MMM_list)
    RSST_somme_component_change_ALPHA_MMM_MMM = np.nanmean(RSST_somme_component_change_ALPHA_MMM, axis=0)
    RSST_somme_component_change_ALPHA_MMM_STD = np.std(RSST_somme_component_change_ALPHA_MMM, axis=0)
    
def calculate_relative_variables():
    """Calcule les variables relatives"""
    global relative_atm_variable_array, relative_atm_variable_array_MMM
    global Patterns_relative_D0_change_array, Patterns_relative_D0_change_array_MMM, Patterns_relative_D0_change_array_STD
    
    # Les calculs existants pour les variables atmosphériques relatives restent inchangés
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
    relative_atm_variable_array_MMM = np.nanmean(relative_atm_variable_array, axis=1)

    # Nouveaux calculs pour Patterns_relative_D0_change
    Patterns_relative_D0_change_list = []
    for i in range(n):
        Patterns_relative_D0_change = relative_D0_change_array[i] / total_coef_array_MMM
        Patterns_relative_D0_change_list.append(Patterns_relative_D0_change)
    Patterns_relative_D0_change_array = np.array(Patterns_relative_D0_change_list)
    Patterns_relative_D0_change_array_MMM = np.nanmean(Patterns_relative_D0_change_array, axis=0)
    Patterns_relative_D0_change_array_STD = np.std(Patterns_relative_D0_change_array, axis=0)

    # Le reste de la fonction reste inchangé
    # Calcul des variables Patterns
    global Patterns_atm_variable_total_array, Patterns_atm_variable_total_array_MMM
    global Patterns_atm_variable_total_array_MMM_CMIP5, Patterns_atm_variable_total_array_MMM_CMIP6
    
    Patterns_atm_variable_total_list = []
    for j in range(5):
        relative_var = relative_atm_variable_array[j]
        Patterns_atm_variable_list = []
        for i in range(n):
            Patterns_atm_variable = relative_var[i]/total_coef_array_MMM
            Patterns_atm_variable_list.append(Patterns_atm_variable)
        Patterns_atm_variable_total_list.append(Patterns_atm_variable_list)
    Patterns_atm_variable_total_array = np.array(Patterns_atm_variable_total_list)
    Patterns_atm_variable_total_array_MMM = np.nanmean(Patterns_atm_variable_total_array, axis=1)
    Patterns_atm_variable_total_array_MMM_CMIP5 = np.nanmean(Patterns_atm_variable_total_array[:,icmip5,:,:], axis=1)
    Patterns_atm_variable_total_array_MMM_CMIP6 = np.nanmean(Patterns_atm_variable_total_array[:,icmip6,:,:], axis=1)
    


def calculate_atmospheric_mean_states():
    """Calcule les états moyens des variables atmosphériques"""
    global atm_variable_change_mean_state_lat, atm_variable_change_mean_state
    global atm_variable_change_mean_state_lat_PACIFIC, atm_variable_change_PACIFIC_mean_state_MMM
    
    atm_variable_change_mean_state_lat = np.nanmean(atm_variable_change[:,:,55:104,0:360], axis=2)
    atm_variable_change_mean_state = np.nanmean(atm_variable_change_mean_state_lat, axis=2)
    
    atm_variable_change_mean_state_lat_PACIFIC = np.nanmean(atm_variable_change[:,:,55:104,160:270], axis=2)
    atm_variable_change_PACIFIC_mean_state_MMM = np.nanmean(atm_variable_change_mean_state_lat_PACIFIC, axis=2)

def calculate_relative_tos_changes():
    """Calcule les changements relatifs de température de surface"""
    global relative_tos_change_array, relative_tos_change_array_MMM, relative_tos_change_array_STD
    
    # mean state calcul
    tos_change_array_fitre_land=np.where(tos_change_array==0,np.nan,tos_change_array)
    tos_change_mean_state_lat=np.nanmean(tos_change_array_fitre_land[:,55:104,0:360],axis=1)
    tos_change_mean_state=np.nanmean(tos_change_mean_state_lat,axis=1)
    
        
    relative_tos_change_list=[]
    for i in range (n):
        relative_tos_change=tos_change_array[i]-tos_change_mean_state[i]
        relative_tos_change_list.append(relative_tos_change)
    relative_tos_change_array=np.array(relative_tos_change_list)
    relative_tos_change_array_MMM=np.nanmean(relative_tos_change_array,axis=0)
    relative_tos_change_array_STD=np.std(relative_tos_change_array,axis=0)




def calculate_relative_D0_changes():
    """Calcule les changements relatifs de D0"""
    global relative_D0_change_array, relative_D0_change_array_MMM
    
    relative_D0_change_list = []
    for i in range(n):
        relative_D0 = D0_change_array[i] - D0_change_array_mean_state[i]
        relative_D0_change_list.append(relative_D0)
    
    relative_D0_change_array = np.array(relative_D0_change_list)
    relative_D0_change_array_MMM = np.nanmean(relative_D0_change_array, axis=0)

def calculate_zhang_mean_states():
    """Calcule les états moyens pour les variables de Zhang"""
    global tos_Zhang_change_mean_state, tos_Zhang_change_mean_state_MMM
    
    tos_Zhang_change_mean_state_lat = np.nanmean(tos_Zhang_change_array[:,55:104,0:360], axis=1)
    tos_Zhang_change_mean_state = np.nanmean(tos_Zhang_change_mean_state_lat, axis=1)
    tos_Zhang_change_mean_state_MMM = np.nanmean(tos_Zhang_change_mean_state, axis=0)

def calculate_relative_coefficients():
    """Calcule les coefficients relatifs"""
    global relative_total_coef_array, relative_total_coef_array_MMM
    
    relative_total_coef_list = []
    for i in range(n):
        relative_total_coef = total_coef_array[i] - total_coef_array_mean_state[i]
        relative_total_coef_list.append(relative_total_coef)
    relative_total_coef_array = np.array(relative_total_coef_list)
    relative_total_coef_array_MMM = np.nanmean(relative_total_coef_array, axis=0)

def run_all_calculations():
    """Main function that executes all calculations in the correct order:
    1. Data loading
    2. Basic coefficient calculations
    3. Atmospheric and heat flux processing
    4. Wind stress analysis
    5. Temperature pattern analysis
    6. Spatial heterogeneity calculations"""
    
    global results, model_list
    results, model_list = load_and_process_data()
    
    init()
    
    

    
    # Execute all calculation functions in proper sequence
     
    

    calculate_LW_coefficients()
    calculate_latent_heat()
    calculate_total_coefficients()
    
    calculate_atmospheric_variables()
    calculate_D0_changes()
    calculate_mean_state_patterns()
    calculate_atmospheric_mean_states()
    
    calculate_relative_D0_changes()
    
    calculate_relative_variables()
    calculate_fractions()
    
    calculate_wind_stress()
    calculate_wind_components()
    calculate_normalized_wind()
    
    calculate_relative_tos_changes()

    calculate_relative_coefficients()
    calculate_spatial_heterogeneity()
    
    calculate_zhang_variables()
    calculate_zhang_mean_states()
    
    calculate_additional_patterns()

    calculate_pattern_sum()
    # calculate_tos_bias()

    print("Tous les calculs ont été effectués avec succès")
    
    print("All calculations completed successfully")

if __name__ == "__main__":
    run_all_calculations()
    print("Models used:", model_list)
