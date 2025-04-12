#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 12:22:04 2025

@author: vincent
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 18:42:02 2025

@author: vincent
"""


import pandas as pd
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import matplotlib.ticker as mticker
import cartopy.feature as cfeature



########################################################################################### histo




TABLE_CMIPS_SOUTH_array = np.array([
    
     RSST_somme_component_CMIPS_SOUTH_change_mean,
     Patterns_relative_D0_CMIPS_SOUTH_change_mean,
     oceanic_flux_deviation_fraction_CMIPS_SOUTH_change_mean,
     Pattern_sum_atm_component_fraction_CMIPS_SOUTH_change_mean,
     Patterns_atm_variable_total_CMIPS_SOUTH_change_array_mean[2],
     Patterns_atm_variable_total_CMIPS_SOUTH_change_array_mean[3],
     Patterns_atm_variable_total_CMIPS_SOUTH_change_array_mean[0],
     Patterns_atm_variable_total_CMIPS_SOUTH_change_array_mean[1]
    
 ]).T




name_DF_list = [
    r"$\delta T_{S}'$",
    r"$\frac{\delta D_{0}'}{\alpha}$",
    r"$\frac{<\delta T_{s}>\alpha'}{\alpha}$",
    r"$\frac{H_{\text{atm}}'}{\alpha}$",
    r"$\frac{\delta Q_{LW}^{dw}'}{\alpha}$",
    r"$\frac{\delta Q_{lh}^{a}'}{\alpha}$",
    r"$\frac{\delta Q_{SW}'}{\alpha}$",
    r"$\frac{\delta Q_{sh}'}{\alpha}$"
]

Df_mean_components_CMIPS_PATTERNS_SOUTH = pd.DataFrame(
     TABLE_CMIPS_SOUTH_array, index=model_list, columns=name_DF_list
 )


RSST_SOUTH_change_mean_var=np.nanvar(RSST_somme_component_CMIPS_SOUTH_change_mean)




TABLE_VARIANCE_all_SOUTH_array = np.array([
    RSST_somme_component_CMIPS_SOUTH_change_mean,
    Patterns_relative_D0_CMIPS_SOUTH_change_mean,
    oceanic_flux_deviation_fraction_CMIPS_SOUTH_change_mean,
    Patterns_atm_variable_total_CMIPS_SOUTH_change_array_mean[2],
    Patterns_atm_variable_total_CMIPS_SOUTH_change_array_mean[3],
    Patterns_atm_variable_total_CMIPS_SOUTH_change_array_mean[0],
    Patterns_atm_variable_total_CMIPS_SOUTH_change_array_mean[1]
   
]).T


# Calcul de TABLE_VARIANCE_all_SOUTH_array_MMM
TABLE_VARIANCE_all_SOUTH_array_MMM = np.nanmean(TABLE_VARIANCE_all_SOUTH_array, axis=0)

# Calcul des écarts
TABLE_VARIANCE_all_SOUTH_array_ecart = TABLE_VARIANCE_all_SOUTH_array - TABLE_VARIANCE_all_SOUTH_array_MMM

# Calcul des covariances en utilisant un seul appel à sum()
TABLE_VARIANCE_all_SOUTH_array_covariance = np.sum(
    RSST_somme_component_CMIPS_SOUTH_change_mean[:, np.newaxis] * TABLE_VARIANCE_all_SOUTH_array_ecart / n,
    axis=0)




######################


TABLE_VARIANCE_sum_SOUTH_array = np.array([
    RSST_somme_component_CMIPS_SOUTH_change_mean,
    Patterns_relative_D0_CMIPS_SOUTH_change_mean,
    oceanic_flux_deviation_fraction_CMIPS_SOUTH_change_mean,
    Pattern_sum_atm_component_fraction_CMIPS_SOUTH_change_mean

   
]).T


# Calcul de TABLE_VARIANCE_sum_SOUTH_array_MMM
TABLE_VARIANCE_sum_SOUTH_array_MMM = np.nanmean(TABLE_VARIANCE_sum_SOUTH_array, axis=0)

# Calcul des écarts
TABLE_VARIANCE_sum_SOUTH_array_ecart = TABLE_VARIANCE_sum_SOUTH_array - TABLE_VARIANCE_sum_SOUTH_array_MMM

# Calcul des covariances en utilisant un seul appel à sum()
TABLE_VARIANCE_sum_SOUTH_array_covariance = np.sum(
    RSST_somme_component_CMIPS_SOUTH_change_mean[:, np.newaxis] * TABLE_VARIANCE_sum_SOUTH_array_ecart / n,
    axis=0)


###########################



data_SOUTH= {

    "RSST": TABLE_VARIANCE_all_SOUTH_array_covariance[0],
    "DO": TABLE_VARIANCE_all_SOUTH_array_covariance[1],

    "A-SST feedback":TABLE_VARIANCE_all_SOUTH_array_covariance[2],
    " total atm": TABLE_VARIANCE_sum_SOUTH_array_covariance[3],
    " LW DOWN": TABLE_VARIANCE_all_SOUTH_array_covariance[3],
    "atm latent": TABLE_VARIANCE_all_SOUTH_array_covariance[4],
    "net Shortwave": TABLE_VARIANCE_all_SOUTH_array_covariance[5],
    "sensible": TABLE_VARIANCE_all_SOUTH_array_covariance[6],
    

}


df_rsst_SOUTH_covariance_equation = pd.DataFrame(data_SOUTH, index=[0])

# Utilisation de Seaborn avec melt pour spécifier les noms de colonnes sur l'axe x

df_melted = df_rsst_SOUTH_covariance_equation.melt(var_name='Variable', value_name='Variance')




TABLE_CMIPS_WEST_array = np.array([
     RSST_somme_component_CMIPS_WEST_change_mean,
     Patterns_relative_D0_CMIPS_WEST_change_mean,
     oceanic_flux_deviation_fraction_CMIPS_WEST_change_mean,
     Pattern_sum_atm_component_fraction_CMIPS_WEST_change_mean,
     Patterns_atm_variable_total_CMIPS_WEST_change_array_mean[2],
     Patterns_atm_variable_total_CMIPS_WEST_change_array_mean[3],
     Patterns_atm_variable_total_CMIPS_WEST_change_array_mean[0],
     Patterns_atm_variable_total_CMIPS_WEST_change_array_mean[1]
    
 ]).T

Df_mean_components_CMIPS_PATTERNS_WEST = pd.DataFrame(
     TABLE_CMIPS_WEST_array, index=model_list, columns=name_DF_list
 )


RSST_WEST_change_mean_var=np.nanvar(RSST_somme_component_CMIPS_WEST_change_mean)




TABLE_VARIANCE_all_WEST_array = np.array([
    RSST_somme_component_CMIPS_WEST_change_mean,
    Patterns_relative_D0_CMIPS_WEST_change_mean,
    oceanic_flux_deviation_fraction_CMIPS_WEST_change_mean,
    Patterns_atm_variable_total_CMIPS_WEST_change_array_mean[2],
    Patterns_atm_variable_total_CMIPS_WEST_change_array_mean[3],
    Patterns_atm_variable_total_CMIPS_WEST_change_array_mean[0],
    Patterns_atm_variable_total_CMIPS_WEST_change_array_mean[1]
   
]).T


# Calcul de TABLE_VARIANCE_all_WEST_array_MMM
TABLE_VARIANCE_all_WEST_array_MMM = np.nanmean(TABLE_VARIANCE_all_WEST_array, axis=0)

# Calcul des écarts
TABLE_VARIANCE_all_WEST_array_ecart = TABLE_VARIANCE_all_WEST_array - TABLE_VARIANCE_all_WEST_array_MMM

# Calcul des covariances en utilisant un seul appel à sum()
TABLE_VARIANCE_all_WEST_array_covariance = np.sum(
    RSST_somme_component_CMIPS_WEST_change_mean[:, np.newaxis] * TABLE_VARIANCE_all_WEST_array_ecart / n,
    axis=0)




######################


TABLE_VARIANCE_sum_WEST_array = np.array([
    RSST_somme_component_CMIPS_WEST_change_mean,
    Patterns_relative_D0_CMIPS_WEST_change_mean,
    oceanic_flux_deviation_fraction_CMIPS_WEST_change_mean,
    Pattern_sum_atm_component_fraction_CMIPS_WEST_change_mean

   
]).T


# Calcul de TABLE_VARIANCE_sum_WEST_array_MMM
TABLE_VARIANCE_sum_WEST_array_MMM = np.nanmean(TABLE_VARIANCE_sum_WEST_array, axis=0)

# Calcul des écarts
TABLE_VARIANCE_sum_WEST_array_ecart = TABLE_VARIANCE_sum_WEST_array - TABLE_VARIANCE_sum_WEST_array_MMM

# Calcul des covariances en utilisant un seul appel à sum()
TABLE_VARIANCE_sum_WEST_array_covariance = np.sum(
    RSST_somme_component_CMIPS_WEST_change_mean[:, np.newaxis] * TABLE_VARIANCE_sum_WEST_array_ecart / n,
    axis=0)


###########################












data_WEST= {

    "RSST": TABLE_VARIANCE_all_WEST_array_covariance[0],
    "DO": TABLE_VARIANCE_all_WEST_array_covariance[1],

    "A-SST feedback":TABLE_VARIANCE_all_WEST_array_covariance[2],
    " total atm": TABLE_VARIANCE_sum_WEST_array_covariance[3],
    " LW DOWN": TABLE_VARIANCE_all_WEST_array_covariance[3],
    "atm latent": TABLE_VARIANCE_all_WEST_array_covariance[4],
    "net Shortwave": TABLE_VARIANCE_all_WEST_array_covariance[5],
    "sensible": TABLE_VARIANCE_all_WEST_array_covariance[6],
    

}


df_rsst_WEST_covariance_equation = pd.DataFrame(data_WEST, index=[0])

# Utilisation de Seaborn avec melt pour spécifier les noms de colonnes sur l'axe x

df_melted_WEST = df_rsst_WEST_covariance_equation.melt(var_name='Variable', value_name='Variance')








#######################################################################################################" EAST #############################################"






TABLE_CMIPS_EAST_array = np.array([
     RSST_somme_component_CMIPS_EAST_change_mean,
     Patterns_relative_D0_CMIPS_EAST_change_mean,
     oceanic_flux_deviation_fraction_CMIPS_EAST_change_mean,
     Pattern_sum_atm_component_fraction_CMIPS_EAST_change_mean,
     Patterns_atm_variable_total_CMIPS_EAST_change_array_mean[2],
     Patterns_atm_variable_total_CMIPS_EAST_change_array_mean[3],
     Patterns_atm_variable_total_CMIPS_EAST_change_array_mean[0],
     Patterns_atm_variable_total_CMIPS_EAST_change_array_mean[1]
    
 ]).T

Df_mean_components_CMIPS_PATTERNS_EAST = pd.DataFrame(
     TABLE_CMIPS_EAST_array, index=model_list, columns=name_DF_list
 )


RSST_EAST_change_mean_var=np.nanvar(RSST_somme_component_CMIPS_EAST_change_mean)




TABLE_VARIANCE_all_EAST_array = np.array([
    RSST_somme_component_CMIPS_EAST_change_mean,
    Patterns_relative_D0_CMIPS_EAST_change_mean,
    oceanic_flux_deviation_fraction_CMIPS_EAST_change_mean,
    Patterns_atm_variable_total_CMIPS_EAST_change_array_mean[2],
    Patterns_atm_variable_total_CMIPS_EAST_change_array_mean[3],
    Patterns_atm_variable_total_CMIPS_EAST_change_array_mean[0],
    Patterns_atm_variable_total_CMIPS_EAST_change_array_mean[1]
   
]).T


# Calcul de TABLE_VARIANCE_all_EAST_array_MMM
TABLE_VARIANCE_all_EAST_array_MMM = np.nanmean(TABLE_VARIANCE_all_EAST_array, axis=0)

# Calcul des écarts
TABLE_VARIANCE_all_EAST_array_ecart = TABLE_VARIANCE_all_EAST_array - TABLE_VARIANCE_all_EAST_array_MMM

# Calcul des covariances en utilisant un seul appel à sum()
TABLE_VARIANCE_all_EAST_array_covariance = np.sum(
    RSST_somme_component_CMIPS_EAST_change_mean[:, np.newaxis] * TABLE_VARIANCE_all_EAST_array_ecart / n,
    axis=0)




######################


TABLE_VARIANCE_sum_EAST_array = np.array([
    RSST_somme_component_CMIPS_EAST_change_mean,
    Patterns_relative_D0_CMIPS_EAST_change_mean,
    oceanic_flux_deviation_fraction_CMIPS_EAST_change_mean,
    Pattern_sum_atm_component_fraction_CMIPS_EAST_change_mean

   
]).T


# Calcul de TABLE_VARIANCE_sum_EAST_array_MMM
TABLE_VARIANCE_sum_EAST_array_MMM = np.nanmean(TABLE_VARIANCE_sum_EAST_array, axis=0)

# Calcul des écarts
TABLE_VARIANCE_sum_EAST_array_ecart = TABLE_VARIANCE_sum_EAST_array - TABLE_VARIANCE_sum_EAST_array_MMM

# Calcul des covariances en utilisant un seul appel à sum()
TABLE_VARIANCE_sum_EAST_array_covariance = np.sum(
    RSST_somme_component_CMIPS_EAST_change_mean[:, np.newaxis] * TABLE_VARIANCE_sum_EAST_array_ecart / n,
    axis=0)


###########################










data_EAST= {

    "RSST": TABLE_VARIANCE_all_EAST_array_covariance[0],
    "DO": TABLE_VARIANCE_all_EAST_array_covariance[1],

    "A-SST feedback":TABLE_VARIANCE_all_EAST_array_covariance[2],
    " total atm": TABLE_VARIANCE_sum_EAST_array_covariance[3],
    " LW DOWN": TABLE_VARIANCE_all_EAST_array_covariance[3],
    "atm latent": TABLE_VARIANCE_all_EAST_array_covariance[4],
    "net Shortwave": TABLE_VARIANCE_all_EAST_array_covariance[5],
    "sensible": TABLE_VARIANCE_all_EAST_array_covariance[6],
    

}


df_rsst_EAST_covariance_equation = pd.DataFrame(data_EAST, index=[0])

# Utilisation de Seaborn avec melt pour spécifier les noms de colonnes sur l'axe x

df_melted_EAST = df_rsst_EAST_covariance_equation.melt(var_name='Variable', value_name='Variance')




RSST_somme_component_CMIPS_GRADIENT_change_mean=RSST_somme_component_CMIPS_EAST_change_mean - RSST_somme_component_CMIPS_WEST_change_mean

TABLE_VARIANCE_all_GRADIENT_array=TABLE_VARIANCE_all_EAST_array - TABLE_VARIANCE_all_WEST_array



# Calcul de TABLE_VARIANCE_all_GRADIENT_array_MMM
TABLE_VARIANCE_all_GRADIENT_array_MMM = np.nanmean(TABLE_VARIANCE_all_GRADIENT_array, axis=0)

# Calcul des écarts
TABLE_VARIANCE_all_GRADIENT_array_ecart = TABLE_VARIANCE_all_GRADIENT_array - TABLE_VARIANCE_all_GRADIENT_array_MMM

# Calcul des covariances en utilisant un seul appel à sum()
TABLE_VARIANCE_all_GRADIENT_array_covariance = np.sum(
    RSST_somme_component_CMIPS_GRADIENT_change_mean[:, np.newaxis] * TABLE_VARIANCE_all_GRADIENT_array_ecart / n,
    axis=0)



TABLE_VARIANCE_sum_GRADIENT_array = TABLE_VARIANCE_sum_EAST_array - TABLE_VARIANCE_sum_WEST_array


# Calcul de TABLE_VARIANCE_sum_GRADIENT_array_MMM
TABLE_VARIANCE_sum_GRADIENT_array_MMM = np.nanmean(TABLE_VARIANCE_sum_GRADIENT_array, axis=0)

# Calcul des écarts
TABLE_VARIANCE_sum_GRADIENT_array_ecart = TABLE_VARIANCE_sum_GRADIENT_array - TABLE_VARIANCE_sum_GRADIENT_array_MMM

# Calcul des covariances en utilisant un seul appel à sum()
TABLE_VARIANCE_sum_GRADIENT_array_covariance = np.sum(
    RSST_somme_component_CMIPS_GRADIENT_change_mean[:, np.newaxis] * TABLE_VARIANCE_sum_GRADIENT_array_ecart / n,
    axis=0)


###########################









data_GRADIENT= {

    "RSST": TABLE_VARIANCE_all_GRADIENT_array_covariance[0],
    "DO": TABLE_VARIANCE_all_GRADIENT_array_covariance[1],

    "A-SST feedback":TABLE_VARIANCE_all_GRADIENT_array_covariance[2],
    " total atm": TABLE_VARIANCE_sum_GRADIENT_array_covariance[3],
    " LW DOWN": TABLE_VARIANCE_all_GRADIENT_array_covariance[3],
    "atm latent": TABLE_VARIANCE_all_GRADIENT_array_covariance[4],
    "net Shortwave": TABLE_VARIANCE_all_GRADIENT_array_covariance[5],
    "sensible": TABLE_VARIANCE_all_GRADIENT_array_covariance[6],
    

}


df_rsst_GRADIENT_covariance_equation = pd.DataFrame(data_GRADIENT, index=[0])

# Utilisation de Seaborn avec melt pour spécifier les noms de colonnes sur l'axe x

df_melted_GRADIENT = df_rsst_GRADIENT_covariance_equation.melt(var_name='Variable', value_name='Variance')

Df_mean_components_CMIPS_PATTERNS_GRADIENT = Df_mean_components_CMIPS_PATTERNS_EAST - Df_mean_components_CMIPS_PATTERNS_WEST





name_DF_list = [
    r"$\mathbf{\Delta_{pres} T'}$",
    r"$\mathbf{\Delta_{pres} O'}$",
    r"$\mathbf{\overline{\Delta T}\alpha'}$",
    r"$\mathbf{\Delta_{pres} Q_{For}'}$",
    r"$\mathbf{\Delta_{pres} LWD'}$",
    r"$\mathbf{\Delta_{pres} LH_{For}'}$",
    r"$\mathbf{\Delta_{pres} SW'}$",
    r"$\mathbf{\Delta_{pres} SH'}$"]
 
####################################################" figure 9  

# Vérifier si les variables COBE sont définies
import numpy as np
import sys

print("Vérification des variables COBE...")
if 'relative_tos_COBE_change_array' not in globals():
    print("La variable relative_tos_COBE_change_array n'est pas définie.")
    
    # Tentative de recréation depuis les résultats si disponibles
    if 'results' in globals() and 'climate_data' in results and 'results' in results['climate_data']:
        if 'tos' in results['climate_data']['results'] and 'relative_tos_COBE_change_array' in results['climate_data']['results']['tos']:
            print("Récupération depuis results...")
            relative_tos_COBE_change_array = results['climate_data']['results']['tos']['relative_tos_COBE_change_array']
            print("Variable récupérée avec succès.")
        else:
            print("Impossible de trouver la variable dans les résultats.")
            # Créer une variable factice
            print("Création d'une variable factice pour déboguer...")
            relative_tos_COBE_change_array = np.zeros((180, 360))
            print("Variable factice créée.")
    else:
        print("Les résultats ne sont pas disponibles ou ne contiennent pas la structure attendue.")
        # Créer une variable factice
        print("Création d'une variable factice pour déboguer...")
        relative_tos_COBE_change_array = np.zeros((180, 360))
        print("Variable factice créée.")
else:
    print("La variable relative_tos_COBE_change_array est déjà définie.")

# Vérification après création
print("Forme de relative_tos_COBE_change_array:", relative_tos_COBE_change_array.shape)


relative_tos_COBE_change_array_EAST=relative_tos_COBE_change_array[78:82,220:270]
relative_tos_COBE_change_array_EAST_MEAN=np.nanmean(relative_tos_COBE_change_array_EAST)



relative_tos_COBE_change_array_WEST=relative_tos_COBE_change_array[78:82,150:200]
relative_tos_COBE_change_array_WEST_MEAN=np.nanmean(relative_tos_COBE_change_array_WEST)

# Définition des noms de colonnes et couleurs
name_DF_list_boxplot = [
    r"$\mathbf{\Delta_{pres} T'}$",
   r"$\mathbf{\Delta_{pres} O'^*}$",
   r"$\mathbf{\overline{\Delta_{pres} T}\alpha'*}$",  # Celui-ci reste inchangé
   r"$\mathbf{\Delta_{pres} Q_{For}'^*}$", 
   r"$\mathbf{\Delta_{pres} LWD'^*}$",
   r"$\mathbf{\Delta_{pres} LH_{For}'^*}$",
   r"$\mathbf{\Delta_{pres} SW'^*}$",
   r"$\mathbf{\Delta_{pres} SH'^*}$"
]

name_DF_list_bar = [
   r"$\mathbf{\Delta_{pres} O'^*}$",
   r"$\mathbf{\overline{\Delta_{pres} T}\alpha'*}$",  # Celui-ci reste inchangé
   r"$\mathbf{\Delta_{pres} Q_{For}'^*}$",
   r"$\mathbf{\Delta_{pres} LWD'^*}$", 
   r"$\mathbf{\Delta_{pres} LH_{For}'^*}$",
   r"$\mathbf{\Delta_{pres} SW'^*}$",
   r"$\mathbf{\Delta_{pres} SH'^*}$"
]
# Configuration des couleurs
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
n_shades = 3
color_shades = [plt.cm.Reds(i/n_shades) for i in range(1, n_shades+1)]
all_colors_boxplot = colors[0:4] + color_shades[0:3] + [color_shades[-1]]
all_colors_bar = colors[1:] + color_shades + ['#8B4513']

# Définir les limites globales pour les deux types de graphiques
global_y_min_boxplot = -0.6  # Pour les boxplots (première ligne)
global_y_max_boxplot = 0.6  # Pour les boxplots (première ligne)
global_y_min_bar = -80    # Pour les barplots (deuxième ligne)
global_y_max_bar = 110     # Pour les barplots (deuxième ligne)



colors = [ '#ff7f0e', '#2ca02c', '#d62728']  # 4 premières couleurs différentes
base_color = '#d62728'
n_shades = 3  # Réduit à 3 au lieu de 4
color_shades = [plt.cm.Reds(i/n_shades) for i in range(1, n_shades+1)]
all_colors = colors + color_shades + ['#8B4513']  # Ajout d'une couleur marron pour l'avant-dernière barre

def convert_to_percentages(data):
    rsst_variance = data['RSST']
    return {key: (value / rsst_variance) * 100 for key, value in data.items() if key != 'RSST'}


# Conversion des données (inchangée)
data_WEST_percentage = convert_to_percentages(data_WEST)
data_EAST_percentage = convert_to_percentages(data_EAST)
data_GRADIENT_percentage = convert_to_percentages(data_GRADIENT)

df_rsst_WEST_percentage = pd.DataFrame(data_WEST_percentage, index=[0])
df_rsst_EAST_percentage = pd.DataFrame(data_EAST_percentage, index=[0])
df_rsst_GRADIENT_percentage = pd.DataFrame(data_GRADIENT_percentage, index=[0])

df_melted_WEST_percentage = df_rsst_WEST_percentage.melt(var_name='Variable', value_name='Percentage')
df_melted_EAST_percentage = df_rsst_EAST_percentage.melt(var_name='Variable', value_name='Percentage')
df_melted_GRADIENT_percentage = df_rsst_GRADIENT_percentage.melt(var_name='Variable', value_name='Percentage')

# Calculer les limites y globales
all_percentages = pd.concat([df_melted_WEST_percentage['Percentage'], 
                             df_melted_EAST_percentage['Percentage'], 
                             df_melted_GRADIENT_percentage['Percentage']])

# Création de la figure principale
fig = plt.figure(figsize=(40, 25), facecolor='white')
gs = gridspec.GridSpec(2, 2)
fig.subplots_adjust(hspace=0.4, wspace=0.2)

# Fonction pour les boxplots
def create_subplot_boxplot(gs_pos, data, title, mean_value=None):
    ax = plt.subplot(gs[gs_pos])
    sns.boxplot(data=data, ax=ax, palette=all_colors_boxplot)
    
    # Définir les limites de l'axe y
    ax.set_ylim(global_y_min_boxplot, global_y_max_boxplot)
    
    # Ajuster les graduations de l'axe y
    y_ticks = np.arange(global_y_min_boxplot, global_y_max_boxplot + 0.2, 0.2)  # Graduations tous les 0.5
    ax.set_yticks(y_ticks)
    
    ax.axhline(y=0, color='black', linewidth=2, linestyle='--')
    ax.vlines(x=[3.5], ymin=global_y_min_boxplot, ymax=global_y_max_boxplot, 
              color='red', linestyle='solid', linewidth=5)
    
    if mean_value is not None:
        ax.plot(0, mean_value, marker='*', color='red', markersize=20, zorder=5)
    
    ax.tick_params(axis='y', labelsize=40)
    ax.set_ylabel("°C", fontsize=40)
    ax.set_xticklabels(name_DF_list_boxplot, fontsize=40, fontweight='bold',rotation=40)
    ax.set_title(title, fontsize=50, fontweight='bold',pad=20)
    ax.set_facecolor('white')
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

# Fonction pour les barplots
def create_subplot_percentage(gs_pos, df_melted, title):
    ax = plt.subplot(gs[gs_pos])
    sns.barplot(x='Variable', y='Percentage', data=df_melted, ax=ax, palette=all_colors_bar)
    
    # Définir les limites de l'axe y
    ax.set_ylim(global_y_min_bar, global_y_max_bar)
    
    # Ajuster les graduations de l'axe y
    y_ticks = np.arange(global_y_min_bar, global_y_max_bar + 20, 20)
    ax.set_yticks(y_ticks)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.vlines(x=[2.5], ymin=global_y_min_bar, ymax=120, 
              color='red', linestyle='solid', linewidth=5)
    
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        if np.isnan(height):
            continue
        y_pos = height + 0.02*(global_y_max_bar - global_y_min_bar) if height > 0 else height - 0.02*(global_y_max_bar - global_y_min_bar)
        va = 'bottom' if height > 0 else 'top'
        ax.text(p.get_x() + p.get_width() / 2., y_pos, f'{height:.1f}%',
                ha='center', va=va, color='black', fontweight='bold', fontsize=35)
    
    ax.tick_params(axis='y', labelsize=40)
    ax.set_ylabel("%", fontsize=40)
    ax.set_xticklabels(name_DF_list_bar, fontsize=40, fontweight='bold',rotation=45, ha="center")
    ax.set_title(title, fontsize=50, fontweight='bold',pad=20)
    ax.set_facecolor('white')
    ax.set_xlabel('')
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

# Création des subplots
# Première ligne : boxplots
create_subplot_boxplot((0, 0), Df_mean_components_CMIPS_PATTERNS_WEST, "(a) WEP", 
                      mean_value=relative_tos_COBE_change_array_WEST_MEAN)
create_subplot_boxplot((0, 1), Df_mean_components_CMIPS_PATTERNS_EAST, "(b) EEP", 
                      mean_value=relative_tos_COBE_change_array_EAST_MEAN)

# Deuxième ligne : barplots
create_subplot_percentage((1, 0), df_melted_WEST_percentage, "(c) WEP")
create_subplot_percentage((1, 1), df_melted_EAST_percentage, "(d) EEP")

plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05)
fig.savefig('your_path/figure_9.pdf', format='pdf', bbox_inches='tight')
fig.savefig('your_path/figure_9.png', format='png',  dpi=100, bbox_inches='tight')

plt.show()




##################################### figure 10



model_list_array = [
    'ACCESS-CM2', 'ACCESS-ESM1-5', 'ACCESS1-0', 'ACCESS1.3', 'AWI-CM-1-1-MR', 'CAMS-CSM1-0', 'CAS-ESM2-0', 'CCSM4',
    'CESM2-WACCM', 'CESM2', 'CIESM', 'CMCC-CESM', 'CMCC-CM2-SR5', 'CMCC-CMS', 'CMCC-CM', 'CMCC-ESM2', 'CNRM-CM5',
    'CNRM-CM6-1-HR', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'CSIRO-Mk3-6-0', 'CanESM2', 'CanESM5', 'E3SM-1-1', 'EC-Earth3-CC',
    'EC-Earth3-Veg-LR', 'EC-Earth3-Veg', 'EC-Earth3', 'FGOALS-f3-L', 'FGOALS-g3', 'FIO-ESM-2-0', 'GFDL-CM3', 'GFDL-CM4',
    'GFDL-ESM2G', 'GFDL-ESM2M', 'GFDL-ESM4', 'IPSL-CM5A-LR', 'IPSL-CM5A-MR', 'IPSL-CM5B-LR', 'IPSL-CM6A-LR', 'KIOST-ESM',
    'MIROC-ES2L', 'MIROC-ESM-CHEM', 'MIROC-ESM', 'MIROC5', 'MIROC6', 'MPI-ESM-LR', 'MPI-ESM-MR', 'MPI-ESM1-2-HR',
    'MPI-ESM1-2-LR', 'MRI-CGCM3', 'MRI-ESM1', 'MRI-ESM2-0', 'NESM3', 'NorESM1-ME', 'NorESM1-M', 'NorESM2-LM',
    'NorESM2-MM', 'TaiESM1', 'UKESM1-0-LL', 'bcc-csm1-1-m', 'bcc-csm1-1', 'inmcm4'
]




var_name = [
    r"$\mathbf{\Delta_{pres} T'}$",
    r"$\mathbf{\Delta_{pres} O'*}$",
    r"$\mathbf{\overline{\Delta_{pres} T}\alpha'*}$",  # Pas besoin de * car c'est un terme différent
    r"$\mathbf{\Delta_{pres} Q_{For}'*}$",
    r"$\mathbf{\Delta_{pres} LWD'*}$",
    r"$\mathbf{\Delta_{pres} LH_{For}'*}$",
    r"$\mathbf{\Delta_{pres} SW'*}$",
    r"$\mathbf{\Delta_{pres} SH'*}$"
]





# Création du DataFrame
DataFrame_EAST = pd.DataFrame(TABLE_CMIPS_EAST_array, index=model_list_array, columns=var_name)

# Réinitialisation de l'index pour avoir une colonne 'Model'
DataFrame_EAST = DataFrame_EAST.reset_index().rename(columns={'index': 'Model'})

# Trier le DataFrame par la première variable (vous pouvez changer cela si nécessaire)
DataFrame_EAST_sorted = DataFrame_EAST.sort_values(var_name[0], ascending=False)

# Diviser en deux catégories basées sur la première variable
DataFrame_EAST_positive = DataFrame_EAST_sorted[DataFrame_EAST_sorted[var_name[0]] > 0]
DataFrame_EAST_negative = DataFrame_EAST_sorted[DataFrame_EAST_sorted[var_name[0]] <= 0]

print(DataFrame_EAST_sorted.head())  # Affiche les premières lignes du DataFrame trié
print("\nModèles positifs :")
print(DataFrame_EAST_positive['Model'].tolist())
print("\nModèles négatifs :")
print(DataFrame_EAST_negative['Model'].tolist())

from shapely import geometry
from collections import namedtuple

Region = namedtuple('Region',field_names=['region_name','lonmin','lonmax','latmin','latmax'])


lon_Pacific=np.array(lons_tos_2075_2100[110:300])
lat_Pacific=np.array(lats_tos_2075_2100[50:110] )

relative_tos_pacific = RSST_somme_component_change_ALPHA_MMM[:, 50:110, 110:300]

# Calcul de la moyenne des changements de SST entre les modèles
relative_tos_pacific_mean = np.mean(relative_tos_pacific, axis=0)

# Calcul du nouveau stippling
# Compter combien de modèles ont un changement positif en chaque point
positive_models = np.sum(relative_tos_pacific > 0, axis=0)
total_models = relative_tos_pacific.shape[0]

# Le stippling est vrai là où au moins 80% des modèles sont du même signe
stippling = (positive_models >= 0.75 * total_models) | (positive_models <= 0.25 * total_models)




sub_region_WEST =  Region(
        region_name="WEST_box", ################# 45:60,200:282
        lonmin=-210,
        lonmax=-160,
        latmin=-2,
        latmax=2)
  
geom_WEST = geometry.box(minx=sub_region_WEST.lonmin,maxx=sub_region_WEST.lonmax,miny=sub_region_WEST.latmin,maxy=sub_region_WEST.latmax)


sub_region_EQUATOR =  Region(
        region_name="EQUATOR_box", ################# 45:60,200:282
        lonmin=-210,
        lonmax=-90,
        latmin=-2,
        latmax=2)
  
geom_EQUATOR = geometry.box(minx=sub_region_EQUATOR.lonmin,maxx=sub_region_EQUATOR.lonmax,miny=sub_region_EQUATOR.latmin,maxy=sub_region_EQUATOR.latmax)




sub_region_EAST =  Region(
        region_name="EAST_box", ################# 45:60,200:282
        lonmin=-140,
        lonmax=-90,
        latmin=-2,
        latmax=2)
  
geom_EAST = geometry.box(minx=sub_region_EAST.lonmin,maxx=sub_region_EAST.lonmax,miny=sub_region_EAST.latmin,maxy=sub_region_EAST.latmax)





sub_region_SOUTH =  Region(
        region_name="SOUTH_box", ################# 45:60,200:282
        lonmin=-130,
        lonmax=-80,
        latmin=-22,
        latmax=-12)
  
geom_SOUTH = geometry.box(minx=sub_region_SOUTH.lonmin,maxx=sub_region_SOUTH.lonmax,miny=sub_region_SOUTH.latmin,maxy=sub_region_SOUTH.latmax)

lon_tropic = lons_tos_2075_2100[110:300]
lat_tropic = lats_tos_2075_2100[50:110]

# Définir la figure et l'axe

# Sélection des 10 modèles les plus froids et leurs valeurs ΔT'
cool_models_df = DataFrame_EAST.nsmallest(10, var_name[0])[['Model', var_name[0]]]


# Définir la figure et l'axe avec 3 subplots
fig, axs = plt.subplots(3, 1, figsize=(16, 16), 
                       subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

# Définir les niveaux pour le contour
ccc = 0.8
step = ccc * 0.05
levels = np.arange(-ccc, ccc + step, step)

def setup_map(ax, title):
    # Ajouter les géométries avec les bordures noires
    ax.add_geometries([geom_EAST], crs=ccrs.PlateCarree(), facecolor="none", 
                     edgecolor="black", linewidth=4)
    ax.add_geometries([geom_WEST], crs=ccrs.PlateCarree(), facecolor="none", 
                     edgecolor="black", linewidth=4)

    # Ajouter les lignes de grille
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, 
                     linewidth=2, color='gray', alpha=1, linestyle='--')
    gl.xlocator = mticker.FixedLocator([120, 140, 160, 180, -160, -140, -120, -100, -80])
    gl.ylocator = mticker.FixedLocator([-30, -20, -10, 0, 10, 20, 30])
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 23}
    gl.ylabel_style = {'size': 23}

    # Ajouter les lignes de côte et les terres
    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.LAND, zorder=100, edgecolor="k")
    
    # Ajouter le titre
    ax.set_title(title, fontsize=30, fontweight='bold')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

# Premier subplot: COBE (inversé avec CMIP MMM)
contour1 = axs[0].contourf(lons_tos_2075_2100[110:300], 
                          lats_tos_2075_2100[50:110],
                          relative_tos_COBE_change_array[50:110, 110:300],
                          transform=ccrs.PlateCarree(), 
                          cmap='RdBu_r', 
                          levels=levels, 
                          extend="max")

setup_map(axs[0], r"(a) COBE $\mathbf{\Delta_{pres}T'}$ change")

# Deuxième subplot: CMIP MMM (inversé avec COBE)
contour2 = axs[1].contourf(lons_tos_2075_2100[110:300], 
                          lats_tos_2075_2100[50:110],
                          RSST_somme_component_change_ALPHA_MMM_MMM[50:110, 110:300],
                          transform=ccrs.PlateCarree(), 
                          cmap='RdBu_r', 
                          levels=levels, 
                          extend="both")

setup_map(axs[1], r"(b) CMIP MMM $\mathbf{\Delta_{pres}T'}$ change")

# Ajouter le stippling pour le CMIP MMM plot
lon_mesh, lat_mesh = np.meshgrid(lon_tropic, lat_tropic)
axs[1].scatter(lon_mesh[stippling], lat_mesh[stippling], 
               color='black', s=1, alpha=0.5, transform=ccrs.PlateCarree())

# Troisième subplot: 10 Coolest Models MMM (inchangé)
coolest_indices = cool_models_df.index
coolest_data = RSST_somme_component_change_ALPHA_MMM[coolest_indices]
coolest_mean = np.mean(coolest_data, axis=0)[50:110, 110:300]

contour3 = axs[2].contourf(lons_tos_2075_2100[110:300], 
                          lats_tos_2075_2100[50:110],
                          coolest_mean,
                          transform=ccrs.PlateCarree(), 
                          cmap='RdBu_r', 
                          levels=levels, 
                          extend="both")

# setup_map(axs[2], "(c) 10 Coolest Models MMM ΔT' change")
setup_map(axs[2], r"(c) 10 Coolest Models MMM $\mathbf{\Delta_{pres}T'}$ change")
# Calculer le stippling pour les 10 modèles les plus froids
positive_models_cool = np.sum(coolest_data[:, 50:110, 110:300] > 0, axis=0)
stippling_cool = (positive_models_cool >= 0.75 * 10) | (positive_models_cool <= 0.25 * 10)

# Ajouter le stippling pour le troisième plot
axs[2].scatter(lon_mesh[stippling_cool], lat_mesh[stippling_cool], 
               color='black', s=1, alpha=0.5, transform=ccrs.PlateCarree())

# Ajouter les barres de couleur
for ax, cont in zip(axs, [contour1, contour2, contour3]):
    cbar = plt.colorbar(cont, orientation="vertical", label="°C", 
                       fraction=0.015, pad=0.04, ax=ax)
    cbar.ax.tick_params(labelsize=25)
    cbar.set_label("°C", size=25)

# Afficher les modèles et leurs valeurs
print("\nLes 10 modèles les plus froids (triés par ΔT'):")
for idx, row in cool_models_df.iterrows():
    print(f"{row['Model']}: {row[var_name[0]]:.3f}°C")

# Ajuster l'espacement entre les sous-graphiques
plt.tight_layout()

# Afficher le graphique
plt.show()

relative_tos_COBE_change_array_EAST=relative_tos_COBE_change_array[78:82,220:270]
relative_tos_COBE_change_array_EAST_MEAN=np.nanmean(relative_tos_COBE_change_array_EAST)



relative_tos_COBE_change_array_WEST=relative_tos_COBE_change_array[78:82,150:200]
relative_tos_COBE_change_array_WEST_MEAN=np.nanmean(relative_tos_COBE_change_array_WEST)


# Création du DataFrame
DataFrame_WEST = pd.DataFrame(TABLE_CMIPS_WEST_array, index=model_list_array, columns=var_name)

# Réinitialisation de l'index pour avoir une colonne 'Model'
DataFrame_WEST = DataFrame_WEST.reset_index().rename(columns={'index': 'Model'})

# Trier le DataFrame par la première variable (vous pouvez changer cela si nécessaire)
DataFrame_WEST_sorted = DataFrame_WEST.sort_values(var_name[0], ascending=False)

# Diviser en deux catégories basées sur la première variable
DataFrame_WEST_positive = DataFrame_WEST_sorted[DataFrame_WEST_sorted[var_name[0]] > 0]
DataFrame_WEST_negative = DataFrame_WEST_sorted[DataFrame_WEST_sorted[var_name[0]] <= 0]

print(DataFrame_WEST_sorted.head())  # Affiche les premières lignes du DataFrame trié
print("\nModèles positifs :")
print(DataFrame_WEST_positive['Model'].tolist())
print("\nModèles négatifs :")
print(DataFrame_WEST_negative['Model'].tolist())



# Création du DataFrame
DataFrame_EAST = pd.DataFrame(TABLE_CMIPS_EAST_array, index=model_list_array, columns=var_name)

# Réinitialisation de l'index pour avoir une colonne 'Model'
DataFrame_EAST = DataFrame_EAST.reset_index().rename(columns={'index': 'Model'})

# Trier le DataFrame par la première variable (vous pouvez changer cela si nécessaire)
DataFrame_EAST_sorted = DataFrame_EAST.sort_values(var_name[0], ascending=False)

# Diviser en deux catégories basées sur la première variable
DataFrame_EAST_positive = DataFrame_EAST_sorted[DataFrame_EAST_sorted[var_name[0]] > 0]
DataFrame_EAST_negative = DataFrame_EAST_sorted[DataFrame_EAST_sorted[var_name[0]] <= 0]

print(DataFrame_EAST_sorted.head())  # Affiche les premières lignes du DataFrame trié
print("\nModèles positifs :")
print(DataFrame_EAST_positive['Model'].tolist())
print("\nModèles négatifs :")
print(DataFrame_EAST_negative['Model'].tolist())

# Calculate GRADIENT (EAST - WEST)
TABLE_CMIPS_GRADIENT_array = TABLE_CMIPS_EAST_array - TABLE_CMIPS_WEST_array

# Creation of the GRADIENT DataFrame
DataFrame_GRADIENT = pd.DataFrame(TABLE_CMIPS_GRADIENT_array, index=model_list_array, columns=var_name)

# Reset index to have a 'Model' column
DataFrame_GRADIENT = DataFrame_GRADIENT.reset_index().rename(columns={'index': 'Model'})

# Sort the DataFrame by the first variable
DataFrame_GRADIENT_sorted = DataFrame_GRADIENT.sort_values(var_name[0], ascending=False)

# Divide into two categories based on the first variable
DataFrame_GRADIENT_positive = DataFrame_GRADIENT_sorted[DataFrame_GRADIENT_sorted[var_name[0]] > 0]
DataFrame_GRADIENT_negative = DataFrame_GRADIENT_sorted[DataFrame_GRADIENT_sorted[var_name[0]] <= 0]

# Print the first few lines of the sorted DataFrame
print(DataFrame_GRADIENT_sorted.head())

# Print positive models
print("\nModèles positifs (GRADIENT) :")
print(DataFrame_GRADIENT_positive['Model'].tolist())

# Print negative models
print("\nModèles négatifs (GRADIENT) :")
print(DataFrame_GRADIENT_negative['Model'].tolist())











############################################################# figure 12


# Création et préparation du DataFrame
df_east = pd.DataFrame(TABLE_CMIPS_EAST_array, index=model_list_array, columns=var_name)
df_east = df_east.reset_index().rename(columns={'index': 'Model'})

def create_combined_correlation_plots(df):
    # Création d'une seule figure
    plt.figure(figsize=(8, 8),  dpi=100)  # Taille ajustée pour une seule figure
    
    # Création du subplot unique
    ax1 = plt.subplot(111)  # 111 au lieu de 131 car une seule figure
    ax1.set_facecolor('white')
    
    # Plot : ΔO' vs ΔQ_{For}'
    x1 = df[var_name[1]]
    y1 = df[var_name[3]]
    correlation1 = np.corrcoef(x1, y1)[0, 1]
    slope1 = np.polyfit(x1, y1, 1)[0]
    
    # On utilise var_name pour accéder aux données mais on passe les noms LaTeX pour l'affichage
    plot_single_correlation(ax1, df, x1, y1, 
                          [var_name[1], "$\mathbf{\Delta_{pres}O'*}$"],  # Liste avec [nom_données, nom_affichage]
                          [var_name[3], "$\mathbf{\Delta_{pres}Q_{For}'*}$"],  # Liste avec [nom_données, nom_affichage]
                          correlation1, slope1, -0.8, 0.8, True)
    
    plt.tight_layout()
    plt.show()


def add_dominance_symbol(model_name, df):
    """
    Ajoute un symbole selon la dominance de ΔO' ou ΔQ_{For}'
    * pour ΔO' dominant
    + pour ΔQ_{For}' dominant
    """
    deltaT_prime = df.loc[df['Model'] == model_name, var_name[0]].values[0]  # ΔT'
    d0_prime = df.loc[df['Model'] == model_name, var_name[1]].values[0]  # ΔO' (corrigé)
    For_prime = df.loc[df['Model'] == model_name, var_name[3]].values[0]  # ΔQ_{For}'
    
    # Debug print
    print(f"\nModel: {model_name}")
    print(f"ΔT': {deltaT_prime:.3f}")
    print(f"ΔO': {d0_prime:.3f}")
    print(f"ΔQ_For': {For_prime:.3f}")  # Correction ici, "For" n'est pas une variable
    
    if abs(d0_prime) > abs(For_prime):
        print("→ ΔO' dominant *")
        return "*"
    else:
        print("→ ΔQ_For' dominant +")
        return "+"

def plot_single_correlation(ax, df, x, y, xlabel_info, ylabel_info, correlation, slope, lim_min, lim_max, add_legend=False):
    """
    xlabel_info et ylabel_info sont maintenant des listes contenant [nom_données, nom_affichage]
    """
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='-', color='lightgray', linewidth=0.5, zorder=0)
    
    cool_models_df = df.nsmallest(10, var_name[0])[['Model', var_name[0]]]
    cool_models = cool_models_df['Model'].tolist()
    
    norm = plt.Normalize(df[var_name[0]].min(), df[var_name[0]].max())
    cmap = plt.cm.RdBu_r
    
    for i, model in enumerate(df['Model']):
        dominance = add_dominance_symbol(model, df)
        if 'relative zonal wind stress' in xlabel_info[0]:
            x_val = x[i]
            y_val = df.loc[df['Model'] == model, ylabel_info[0]].values[0]
        else:
            x_val = df.loc[df['Model'] == model, xlabel_info[0]].values[0]
            y_val = df.loc[df['Model'] == model, ylabel_info[0]].values[0]
            
        t_val = df.loc[df['Model'] == model, var_name[0]].values[0]
        color = cmap(norm(t_val))
        
        if model in cool_models:
            rank = cool_models.index(model) + 1
            ax.scatter(x_val, y_val, c=[color], s=200, edgecolor='black', linewidth=1, 
                      marker='*', zorder=3)
            ax.annotate(str(rank), (x_val, y_val), xytext=(-10, 10),
                       textcoords='offset points', fontsize=15,
                       fontweight='bold', color='black', zorder=4)
        else:
            ax.scatter(x_val, y_val, c=[color], s=100, edgecolor='black', linewidth=1, 
                      alpha=0.7, marker='o', zorder=2)
    
    if 'relative zonal wind stress' not in xlabel_info[0]:
        ax.plot([lim_min, lim_max], [lim_max, lim_min], 'k--', alpha=0.5, zorder=1)
    
    xlabel_with_units = f"{xlabel_info[1]} ({'10² N·m⁻²' if 'relative zonal wind stress' in xlabel_info[0] else '°C'})"
    ylabel_with_units = f"{ylabel_info[1]} (°C)"
    
    ax.set_xlabel(xlabel_with_units, fontsize=20)
    ax.set_ylabel(ylabel_with_units, fontsize=20)
    
    ax.set_title(f' {xlabel_info[1]} vs. {ylabel_info[1]}\nr = {correlation:.2f}, slope = {slope:.2f}', 
                 fontsize=25, pad=20)
    
    if add_legend:
        legend_text = 'Markers:\n* : 10 Coolest Models\n○ : Other Models\n---: y=-x\n\nCool models ranked by ΔT\':\n'
        for rank, (model, dt) in enumerate(zip(cool_models, cool_models_df[var_name[0]]), 1):
            dominance = add_dominance_symbol(model, df)
            legend_text += f"{rank}. {model}{dominance} ({dt:.2f})\n"
        
        plt.figtext(1, 0.5, legend_text, fontsize=15,
                   verticalalignment='center',
                   bbox=dict(facecolor='white', edgecolor='black', pad=8))
        
    
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1.0, alpha=0.5, zorder=2)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=1.0, alpha=0.5, zorder=2)
    ax.tick_params(axis='both', which='major', labelsize=15)  # Augmente la taille des nombres sur les axes

    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
    fig.savefig('your_path/figure_12.pdf', format='pdf', bbox_inches='tight')
    fig.savefig('your_path/figure_12.png', format='png',  dpi=100, bbox_inches='tight')

    plt.show()

        
create_combined_correlation_plots(df_east)



###############################" figure 11


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Chargement des données combinées (historiques et futures)
temperature_data_combined = pd.read_csv('/future_temperature_ranking.csv')

# Vérification des données chargées
print("Données combinées chargées :")
print(temperature_data_combined.columns)  # Afficher les noms de colonnes

# Renommage des colonnes en notation mathématique
temperature_data_combined = temperature_data_combined.rename(columns={
    'Temperature_change_East': r"$\mathbf{\Delta_{pres} T'}$",
    'Temperature_change_East_Future': r"$\mathbf{\Delta T'}$"
})

print("Colonnes après renommage:")
print(temperature_data_combined.columns)

# Version modifiée de la fonction pour les médianes
def add_median_lines_and_values(ax, all_median, warmest_median, coolest_median, observed_value=None, title=None):
    """Ajoute les lignes de médiane avec légende en dehors de la figure"""
    
    lines = []
    labels = []
    
    # Pour COBE (uniquement pour ΔT')
    if observed_value is not None:
        l0 = ax.axhline(observed_value, color='black', linestyle='--', linewidth=2, zorder=2, alpha=0.7)
        lines.append(l0)
        labels.append(f'COBE: {observed_value:.2f}')
    
    # Pour la médiane de tous les modèles
    l1 = ax.axhline(all_median, color='green', linestyle='--', linewidth=2, zorder=2, alpha=0.7)
    lines.append(l1)
    labels.append(f'MMM: {all_median:.2f}')
    
    # Pour les modèles chauds
    l2 = ax.axhline(warmest_median, color='red', linestyle='--', linewidth=2, zorder=2, alpha=0.7)
    lines.append(l2)
    labels.append(f'10 warmest: {warmest_median:.2f}')
    
    # Pour les modèles froids
    l3 = ax.axhline(coolest_median, color='blue', linestyle='--', linewidth=2, zorder=2, alpha=0.7)
    lines.append(l3)
    labels.append(f'10 coolest: {coolest_median:.2f}')
    
    # Création de la légende
    leg = ax.legend(lines, labels,
              loc='center left',
              bbox_to_anchor=(1.01, 0.5),
              fontsize=14,
              frameon=True,
              edgecolor='black',
              facecolor='white',
              framealpha=1.0,
              borderaxespad=0.1,
              handlelength=1.5)
    
    # Mettre le texte de la légende en gras
    for text in leg.get_texts():
        text.set_weight('bold')

def add_dominance_symbol(model_name, df):
    """
    Ajoute un symbole selon la dominance de ΔO' ou ΔQ_{For}'
    * pour ΔO' dominant
    + pour ΔQ_{For}' dominant
    """
    deltaT_prime = df.loc[df['Model'] == model_name, var_name[0]].values[0]  # ΔT'
    d0_prime = df.loc[df['Model'] == model_name, var_name[1]].values[0]  # ΔO' (corrigé)
    For_prime = df.loc[df['Model'] == model_name, var_name[3]].values[0]  # ΔQ_{For}'
    
    # Debug print
    print(f"\nModel: {model_name}")
    print(f"ΔT': {deltaT_prime:.3f}")
    print(f"ΔO': {d0_prime:.3f}")
    print(f"ΔQ_For': {For_prime:.3f}")  # Correction ici, "For" n'est pas une variable
    
    if abs(d0_prime) > abs(For_prime):
        print("→ ΔO' dominant *")
        return "*"
    else:
        print("→ ΔQ_For' dominant +")
        return "+"

def plot_variable(ax, df, variable, cmap, observed_delta_t=None, show_labels=False, 
                  y_limit=None, first_variable=False, show_dominance=False, show_model_names=True):
    var_data = df[variable]
    models = df['Model']
    
    norm = Normalize(vmin=y_limit[0], vmax=y_limit[1])
    
    ax.set_facecolor('white')
    ax.grid(True, linestyle='-', linewidth=0.5, color='lightgray', zorder=0)
    
    bar_positions = list(range(len(var_data)))
    bars = ax.bar(bar_positions, var_data, zorder=3)
    
    for bar in bars:
        bar.set_color(cmap(norm(bar.get_height())))
        bar.set_edgecolor('black')
        bar.set_linewidth(1)
    
    # Ajout des zones colorées et lignes verticales
    ax.axvline(x=9.5, color='red', linestyle='--', linewidth=1, zorder=4)
    ax.axvline(x=-0.35, color='red', linestyle='--', linewidth=1, zorder=4)
    ax.axvspan(0, 9.5, facecolor='red', alpha=0.1, zorder=1)
    
    ax.axvline(x=52.5, color='blue', linestyle='--', linewidth=1, zorder=4)
    ax.axvline(x=62.5, color='blue', linestyle='--', linewidth=1, zorder=4)
    ax.axvspan(52.5, 62.5, facecolor='blue', alpha=0.1, zorder=1)
    
    # Calcul des médianes
    all_models_median = df[variable].median()
    warmest_median = df.iloc[:10][variable].median()
    coolest_median = df.iloc[-10:][variable].median()
    
    # Ajout des lignes de médiane et valeurs
    add_median_lines_and_values(ax, all_models_median, warmest_median, coolest_median, 
                          observed_value=observed_delta_t if first_variable else None,
                          title=ax.get_title())
    
    if variable == var_name[4]:
        ax.set_title(f"{variable} [150°W/90°W]", fontsize=14)
    else:
        ax.set_title(f"{variable} [140°W/90°W]", fontsize=14)
    
    ax.set_ylabel(variable, fontsize=15)
    ax.axhline(0, linestyle="-", color="black", linewidth=1, zorder=2)
    
    ax.set_xticks(bar_positions)
    
    if show_labels and show_model_names:
      # Affichage simple des noms de modèles
      ax.set_xticklabels(models, rotation=90, ha='center', fontsize=10, weight='bold')
    else:
      ax.set_xticklabels([])
    ax.set_ylim(y_limit)
    ax.tick_params(axis='y', labelsize=11)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor('black')

def plot_combined_variables_single_column(df, df_combined, title_prefix, observed_delta_t, y_limit):
    plt.rcParams['figure.dpi'] = 300
    
    # Création de la figure plus étroite pour une seule colonne
    fig = plt.figure(figsize=(15, 18))
    
    gs = GridSpec(5, 1, figure=fig, height_ratios=[1, 1, 1, 1, 1], hspace=0.4)
    
    cmap = plt.colormaps['RdBu_r']
    
    # Définir les limites pour la première figure (données futures)
    future_y_limit = (-2, 2.0)
   
    # Premier graphique avec les données futures
    ax = fig.add_subplot(gs[0, 0])
    
    # Utiliser les données dans l'ordre original du CSV sans tri
    plot_variable(ax, df_combined, "$\mathbf{\Delta T'}$", cmap, 
                None, show_labels=False,
                y_limit=future_y_limit,
                first_variable=True,
                show_dominance=False,
                show_model_names=False)
    ax.set_title("$\mathbf{(a)}$ $\mathbf{\Delta T'}$", fontsize=20, pad=10)
    
    # Tri des données historiques par ΔT'
    delta_t_var = var_name[0]
    df_sorted = df.sort_values(delta_t_var, ascending=False)
    
    # Liste des variables restantes à tracer
    plot_vars = [delta_t_var, var_name[1], var_name[2], var_name[3]]
    
    # Lettres pour les sous-figures
    subfig_letters = ['b', 'c', 'd', 'e']
    
    # Tracé des graphiques restants
    for idx, var in enumerate(plot_vars):
        ax = fig.add_subplot(gs[idx+1, 0])
        
        if idx < 3:
            plot_variable(ax, df_sorted, var, cmap, 
                          observed_delta_t if idx == 0 else None, 
                          show_labels=False,  
                          y_limit=y_limit, 
                          first_variable=(idx == 0),
                          show_dominance=False,
                          show_model_names=False)
        else:
            plot_variable(ax, df_sorted, var, cmap, 
                          None, show_labels=True, 
                          y_limit=y_limit, 
                          first_variable=False,
                          show_dominance=True,
                          show_model_names=True)
        
        ax.set_title(f"$\mathbf{{({subfig_letters[idx]})}}$ {var}", fontsize=20, pad=10)
    
    plt.subplots_adjust(right=0.9)
    fig.savefig('your_path/figure_11.pdf', format='pdf', bbox_inches='tight')
    fig.savefig('your_path/figure_11.png', format='png',  dpi=100, bbox_inches='tight')
    
    plt.show()

# Définition des variables nécessaires
var_name = [
    r"$\mathbf{\Delta_{pres} T'}$",
    r"$\mathbf{\Delta_{pres} O'*}$",
    r"$\mathbf{\overline{\Delta_{pres} T}\alpha'*}$",  # Pas besoin de * car c'est un terme différent
    r"$\mathbf{\Delta_{pres} Q_{For}'*}$",
    r"$\mathbf{\Delta_{pres} LWD'*}$",
    r"$\mathbf{\Delta_{pres} LH_{For}'*}$",
    r"$\mathbf{\Delta_{pres} SW'*}$",
    r"$\mathbf{\Delta_{pres} SH'*}$"
]

# Création du DataFrame historique
df_east = pd.DataFrame(TABLE_CMIPS_EAST_array, index=model_list_array, columns=var_name)
df_east = df_east.reset_index().rename(columns={'index': 'Model'})

# Chargement des valeurs observées
relative_tos_COBE_change_array_EAST = relative_tos_COBE_change_array[78:82, 220:270]
relative_tos_COBE_change_array_EAST_MEAN = np.nanmean(relative_tos_COBE_change_array_EAST)
# Appel de la fonction avec les nouvelles données combinées
title_prefix = "[EEP]"
observed_delta_t = relative_tos_COBE_change_array_EAST_MEAN
y_limit = (-0.7, 0.7)
plot_combined_variables_single_column(df_east, temperature_data_combined, title_prefix, observed_delta_t, y_limit)

