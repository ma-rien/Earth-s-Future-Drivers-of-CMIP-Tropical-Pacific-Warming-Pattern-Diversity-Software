#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 11:46:43 2025

@author: vincent
"""



import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from shapely import geometry
from collections import namedtuple
import matplotlib.ticker as mticker
import pandas as pd

import numpy as np

import seaborn as sns
import cartopy.crs as ccrs

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


from scipy.stats import shapiro

from scipy import stats
from matplotlib.lines import Line2D


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


RSST_SOUTH_change_mean_var = np.nanvar(
    RSST_somme_component_CMIPS_SOUTH_change_mean)


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
TABLE_VARIANCE_all_SOUTH_array_MMM = np.nanmean(
    TABLE_VARIANCE_all_SOUTH_array, axis=0)

# Calcul des écarts
TABLE_VARIANCE_all_SOUTH_array_ecart = TABLE_VARIANCE_all_SOUTH_array - \
    TABLE_VARIANCE_all_SOUTH_array_MMM

# Calcul des covariances en utilisant un seul appel à sum()
TABLE_VARIANCE_all_SOUTH_array_covariance = np.sum(
    RSST_somme_component_CMIPS_SOUTH_change_mean[:,
                                                 np.newaxis] * TABLE_VARIANCE_all_SOUTH_array_ecart / n,
    axis=0)


######################


TABLE_VARIANCE_sum_SOUTH_array = np.array([
    RSST_somme_component_CMIPS_SOUTH_change_mean,
    Patterns_relative_D0_CMIPS_SOUTH_change_mean,
    oceanic_flux_deviation_fraction_CMIPS_SOUTH_change_mean,
    Pattern_sum_atm_component_fraction_CMIPS_SOUTH_change_mean


]).T


# Calcul de TABLE_VARIANCE_sum_SOUTH_array_MMM
TABLE_VARIANCE_sum_SOUTH_array_MMM = np.nanmean(
    TABLE_VARIANCE_sum_SOUTH_array, axis=0)

# Calcul des écarts
TABLE_VARIANCE_sum_SOUTH_array_ecart = TABLE_VARIANCE_sum_SOUTH_array - \
    TABLE_VARIANCE_sum_SOUTH_array_MMM

# Calcul des covariances en utilisant un seul appel à sum()
TABLE_VARIANCE_sum_SOUTH_array_covariance = np.sum(
    RSST_somme_component_CMIPS_SOUTH_change_mean[:,
                                                 np.newaxis] * TABLE_VARIANCE_sum_SOUTH_array_ecart / n,
    axis=0)


###########################


data_SOUTH = {

    "RSST": TABLE_VARIANCE_all_SOUTH_array_covariance[0],
    "DO": TABLE_VARIANCE_all_SOUTH_array_covariance[1],

    "A-SST feedback": TABLE_VARIANCE_all_SOUTH_array_covariance[2],
    " total atm": TABLE_VARIANCE_sum_SOUTH_array_covariance[3],
    " LW DOWN": TABLE_VARIANCE_all_SOUTH_array_covariance[3],
    "atm latent": TABLE_VARIANCE_all_SOUTH_array_covariance[4],
    "net Shortwave": TABLE_VARIANCE_all_SOUTH_array_covariance[5],
    "sensible": TABLE_VARIANCE_all_SOUTH_array_covariance[6],


}


df_rsst_SOUTH_covariance_equation = pd.DataFrame(data_SOUTH, index=[0])

# Utilisation de Seaborn avec melt pour spécifier les noms de colonnes sur l'axe x

df_melted = df_rsst_SOUTH_covariance_equation.melt(
    var_name='Variable', value_name='Variance')


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


RSST_WEST_change_mean_var = np.nanvar(
    RSST_somme_component_CMIPS_WEST_change_mean)


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
TABLE_VARIANCE_all_WEST_array_MMM = np.nanmean(
    TABLE_VARIANCE_all_WEST_array, axis=0)

# Calcul des écarts
TABLE_VARIANCE_all_WEST_array_ecart = TABLE_VARIANCE_all_WEST_array - \
    TABLE_VARIANCE_all_WEST_array_MMM

# Calcul des covariances en utilisant un seul appel à sum()
TABLE_VARIANCE_all_WEST_array_covariance = np.sum(
    RSST_somme_component_CMIPS_WEST_change_mean[:,
                                                np.newaxis] * TABLE_VARIANCE_all_WEST_array_ecart / n,
    axis=0)


######################


TABLE_VARIANCE_sum_WEST_array = np.array([
    RSST_somme_component_CMIPS_WEST_change_mean,
    Patterns_relative_D0_CMIPS_WEST_change_mean,
    oceanic_flux_deviation_fraction_CMIPS_WEST_change_mean,
    Pattern_sum_atm_component_fraction_CMIPS_WEST_change_mean


]).T


# Calcul de TABLE_VARIANCE_sum_WEST_array_MMM
TABLE_VARIANCE_sum_WEST_array_MMM = np.nanmean(
    TABLE_VARIANCE_sum_WEST_array, axis=0)

# Calcul des écarts
TABLE_VARIANCE_sum_WEST_array_ecart = TABLE_VARIANCE_sum_WEST_array - \
    TABLE_VARIANCE_sum_WEST_array_MMM

# Calcul des covariances en utilisant un seul appel à sum()
TABLE_VARIANCE_sum_WEST_array_covariance = np.sum(
    RSST_somme_component_CMIPS_WEST_change_mean[:,
                                                np.newaxis] * TABLE_VARIANCE_sum_WEST_array_ecart / n,
    axis=0)


###########################


data_WEST = {

    "RSST": TABLE_VARIANCE_all_WEST_array_covariance[0],
    "DO": TABLE_VARIANCE_all_WEST_array_covariance[1],

    "A-SST feedback": TABLE_VARIANCE_all_WEST_array_covariance[2],
    " total atm": TABLE_VARIANCE_sum_WEST_array_covariance[3],
    " LW DOWN": TABLE_VARIANCE_all_WEST_array_covariance[3],
    "atm latent": TABLE_VARIANCE_all_WEST_array_covariance[4],
    "net Shortwave": TABLE_VARIANCE_all_WEST_array_covariance[5],
    "sensible": TABLE_VARIANCE_all_WEST_array_covariance[6],


}


df_rsst_WEST_covariance_equation = pd.DataFrame(data_WEST, index=[0])

# Utilisation de Seaborn avec melt pour spécifier les noms de colonnes sur l'axe x

df_melted_WEST = df_rsst_WEST_covariance_equation.melt(
    var_name='Variable', value_name='Variance')


# " EAST #############################################"


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


RSST_EAST_change_mean_var = np.nanvar(
    RSST_somme_component_CMIPS_EAST_change_mean)


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
TABLE_VARIANCE_all_EAST_array_MMM = np.nanmean(
    TABLE_VARIANCE_all_EAST_array, axis=0)

# Calcul des écarts
TABLE_VARIANCE_all_EAST_array_ecart = TABLE_VARIANCE_all_EAST_array - \
    TABLE_VARIANCE_all_EAST_array_MMM

# Calcul des covariances en utilisant un seul appel à sum()
TABLE_VARIANCE_all_EAST_array_covariance = np.sum(
    RSST_somme_component_CMIPS_EAST_change_mean[:,
                                                np.newaxis] * TABLE_VARIANCE_all_EAST_array_ecart / n,
    axis=0)


######################


TABLE_VARIANCE_sum_EAST_array = np.array([
    RSST_somme_component_CMIPS_EAST_change_mean,
    Patterns_relative_D0_CMIPS_EAST_change_mean,
    oceanic_flux_deviation_fraction_CMIPS_EAST_change_mean,
    Pattern_sum_atm_component_fraction_CMIPS_EAST_change_mean


]).T


# Calcul de TABLE_VARIANCE_sum_EAST_array_MMM
TABLE_VARIANCE_sum_EAST_array_MMM = np.nanmean(
    TABLE_VARIANCE_sum_EAST_array, axis=0)

# Calcul des écarts
TABLE_VARIANCE_sum_EAST_array_ecart = TABLE_VARIANCE_sum_EAST_array - \
    TABLE_VARIANCE_sum_EAST_array_MMM

# Calcul des covariances en utilisant un seul appel à sum()
TABLE_VARIANCE_sum_EAST_array_covariance = np.sum(
    RSST_somme_component_CMIPS_EAST_change_mean[:,
                                                np.newaxis] * TABLE_VARIANCE_sum_EAST_array_ecart / n,
    axis=0)


###########################


data_EAST = {

    "RSST": TABLE_VARIANCE_all_EAST_array_covariance[0],
    "DO": TABLE_VARIANCE_all_EAST_array_covariance[1],

    "A-SST feedback": TABLE_VARIANCE_all_EAST_array_covariance[2],
    " total atm": TABLE_VARIANCE_sum_EAST_array_covariance[3],
    " LW DOWN": TABLE_VARIANCE_all_EAST_array_covariance[3],
    "atm latent": TABLE_VARIANCE_all_EAST_array_covariance[4],
    "net Shortwave": TABLE_VARIANCE_all_EAST_array_covariance[5],
    "sensible": TABLE_VARIANCE_all_EAST_array_covariance[6],


}


df_rsst_EAST_covariance_equation = pd.DataFrame(data_EAST, index=[0])

# Utilisation de Seaborn avec melt pour spécifier les noms de colonnes sur l'axe x

df_melted_EAST = df_rsst_EAST_covariance_equation.melt(
    var_name='Variable', value_name='Variance')


RSST_somme_component_CMIPS_GRADIENT_change_mean = RSST_somme_component_CMIPS_EAST_change_mean - \
    RSST_somme_component_CMIPS_WEST_change_mean

TABLE_VARIANCE_all_GRADIENT_array = TABLE_VARIANCE_all_EAST_array - \
    TABLE_VARIANCE_all_WEST_array


# Calcul de TABLE_VARIANCE_all_GRADIENT_array_MMM
TABLE_VARIANCE_all_GRADIENT_array_MMM = np.nanmean(
    TABLE_VARIANCE_all_GRADIENT_array, axis=0)

# Calcul des écarts
TABLE_VARIANCE_all_GRADIENT_array_ecart = TABLE_VARIANCE_all_GRADIENT_array - \
    TABLE_VARIANCE_all_GRADIENT_array_MMM

# Calcul des covariances en utilisant un seul appel à sum()
TABLE_VARIANCE_all_GRADIENT_array_covariance = np.sum(
    RSST_somme_component_CMIPS_GRADIENT_change_mean[:,
                                                    np.newaxis] * TABLE_VARIANCE_all_GRADIENT_array_ecart / n,
    axis=0)


TABLE_VARIANCE_sum_GRADIENT_array = TABLE_VARIANCE_sum_EAST_array - \
    TABLE_VARIANCE_sum_WEST_array


# Calcul de TABLE_VARIANCE_sum_GRADIENT_array_MMM
TABLE_VARIANCE_sum_GRADIENT_array_MMM = np.nanmean(
    TABLE_VARIANCE_sum_GRADIENT_array, axis=0)

# Calcul des écarts
TABLE_VARIANCE_sum_GRADIENT_array_ecart = TABLE_VARIANCE_sum_GRADIENT_array - \
    TABLE_VARIANCE_sum_GRADIENT_array_MMM

# Calcul des covariances en utilisant un seul appel à sum()
TABLE_VARIANCE_sum_GRADIENT_array_covariance = np.sum(
    RSST_somme_component_CMIPS_GRADIENT_change_mean[:,
                                                    np.newaxis] * TABLE_VARIANCE_sum_GRADIENT_array_ecart / n,
    axis=0)


###########################


data_GRADIENT = {

    "RSST": TABLE_VARIANCE_all_GRADIENT_array_covariance[0],
    "DO": TABLE_VARIANCE_all_GRADIENT_array_covariance[1],

    "A-SST feedback": TABLE_VARIANCE_all_GRADIENT_array_covariance[2],
    " total atm": TABLE_VARIANCE_sum_GRADIENT_array_covariance[3],
    " LW DOWN": TABLE_VARIANCE_all_GRADIENT_array_covariance[3],
    "atm latent": TABLE_VARIANCE_all_GRADIENT_array_covariance[4],
    "net Shortwave": TABLE_VARIANCE_all_GRADIENT_array_covariance[5],
    "sensible": TABLE_VARIANCE_all_GRADIENT_array_covariance[6],


}


df_rsst_GRADIENT_covariance_equation = pd.DataFrame(data_GRADIENT, index=[0])

# Utilisation de Seaborn avec melt pour spécifier les noms de colonnes sur l'axe x

df_melted_GRADIENT = df_rsst_GRADIENT_covariance_equation.melt(
    var_name='Variable', value_name='Variance')

Df_mean_components_CMIPS_PATTERNS_GRADIENT = Df_mean_components_CMIPS_PATTERNS_EAST - \
    Df_mean_components_CMIPS_PATTERNS_WEST




Region = namedtuple('Region', field_names=[
                    'region_name', 'lonmin', 'lonmax', 'latmin', 'latmax'])


lon_Pacific = np.array(lons_tos_2075_2100[110:300])
lat_Pacific = np.array(lats_tos_2075_2100[50:110])


sub_region_WEST = Region(
    region_name="WEST_box",  # 45:60,200:282
    lonmin=-210,
    lonmax=-160,
    latmin=-2,
    latmax=2)

geom_WEST = geometry.box(minx=sub_region_WEST.lonmin, maxx=sub_region_WEST.lonmax,
                         miny=sub_region_WEST.latmin, maxy=sub_region_WEST.latmax)


sub_region_EQUATOR = Region(
    region_name="EQUATOR_box",  # 45:60,200:282
    lonmin=-210,
    lonmax=-90,
    latmin=-2,
    latmax=2)

geom_EQUATOR = geometry.box(minx=sub_region_EQUATOR.lonmin, maxx=sub_region_EQUATOR.lonmax,
                            miny=sub_region_EQUATOR.latmin, maxy=sub_region_EQUATOR.latmax)


sub_region_EAST = Region(
    region_name="EAST_box",  # 45:60,200:282
    lonmin=-140,
    lonmax=-90,
    latmin=-2,
    latmax=2)

geom_EAST = geometry.box(minx=sub_region_EAST.lonmin, maxx=sub_region_EAST.lonmax,
                         miny=sub_region_EAST.latmin, maxy=sub_region_EAST.latmax)


sub_region_SOUTH = Region(
    region_name="SOUTH_box",  # 45:60,200:282
    lonmin=-130,
    lonmax=-80,
    latmin=-22,
    latmax=-12)

geom_SOUTH = geometry.box(minx=sub_region_SOUTH.lonmin, maxx=sub_region_SOUTH.lonmax,
                          miny=sub_region_SOUTH.latmin, maxy=sub_region_SOUTH.latmax)


################################################ figure 1 ########################

relative_tos_pacific = RSST_somme_component_change_ALPHA_MMM[:,
                                                             50:110, 110:300]

# Calcul de la moyenne des changements de SST entre les modèles
relative_tos_pacific_mean = np.mean(relative_tos_pacific, axis=0)

# Calcul du nouveau stippling
# Compter combien de modèles ont un changement positif en chaque point
positive_models = np.sum(relative_tos_pacific > 0, axis=0)
total_models = relative_tos_pacific.shape[0]

# Le stippling est vrai là où au moins 80% des modèles sont du même signe
stippling = (positive_models >= 0.75 *
             total_models) | (positive_models <= 0.25 * total_models)


lon_tropic = lons_tos_2075_2100[110:300]
lat_tropic = lats_tos_2075_2100[50:110]


# Définir la figure et l'axe
fig, axs = plt.subplots(2, 1, figsize=(16, 11), subplot_kw={
                        'projection': ccrs.PlateCarree(central_longitude=180)})

# Définir les niveaux pour le contour de la première carte
ccc = 2
step = ccc * 0.05
levels = np.arange(-ccc, ccc + step, step)

# Tracer le contour pour la première carte
contour = axs[0].contourf(lons_tos_2075_2100[110:300], lats_tos_2075_2100[50:110], RSST_somme_component_change_ALPHA_MMM_MMM[50:110, 110:300],
                          transform=ccrs.PlateCarree(), cmap='RdBu_r', levels=levels, extend="both")

# Ajouter la barre de couleur pour la première carte
cbar = plt.colorbar(contour, orientation="vertical",
                    label="°C", fraction=0.015, pad=0.04, ax=axs[0])
cbar.ax.tick_params(labelsize=25)
cbar.set_label("°C", size=25)

# Ajout du stippling
lon_mesh, lat_mesh = np.meshgrid(lon_tropic, lat_tropic)
axs[0].scatter(lon_mesh[stippling], lat_mesh[stippling],
               color='black', s=1, alpha=0.5, transform=ccrs.PlateCarree())

# Ajouter les lignes de grille pour la première carte
gl = axs[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=1, linestyle='--')
gl.xlocator = mticker.FixedLocator(
    [120, 140, 160, 180, -160, -140, -120, -100, -80])
gl.ylocator = mticker.FixedLocator([-30, -20, -10, 0, 10, 20, 30])
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 23}
gl.ylabel_style = {'size': 23}

# Ajouter les lignes de côte et les terres pour la première carte
axs[0].coastlines(resolution='50m')
axs[0].add_feature(cfeature.LAND, zorder=100, edgecolor="k")

# Ajouter le titre au premier sous-graphique
axs[0].set_title("(a)  MMM ΔT' change", fontsize=30, fontweight='bold')
for spine in axs[0].spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)  # plt.ylim([-2, 2])

# Définir les niveaux pour le contour de la deuxième carte
ccc = 0.6
step = 0.025
levels = np.arange(0, ccc + step, step)

# Tracer le contour pour la deuxième carte
contour = axs[1].contourf(lons_tos_2075_2100[110:300], lats_tos_2075_2100[50:110], relative_tos_change_array_STD[50:110, 110:300],
                          transform=ccrs.PlateCarree(), cmap='jet', levels=levels, extend="max")

# Ajouter la barre de couleur pour la deuxième carte
cbar = plt.colorbar(contour, orientation="vertical",
                    label="°C", fraction=0.015, pad=0.04, ax=axs[1])
cbar.set_ticks(np.arange(0, 0.7, 0.1))
cbar.ax.tick_params(labelsize=25)
cbar.set_label("°C", size=25)

# Ajouter les géométries avec les bordures noires
axs[1].add_geometries([geom_SOUTH], crs=ccrs.PlateCarree(),
                      facecolor="none", edgecolor="black", linewidth=4)
axs[1].add_geometries([geom_EAST], crs=ccrs.PlateCarree(),
                      facecolor="none", edgecolor="black", linewidth=4)
axs[1].add_geometries([geom_WEST], crs=ccrs.PlateCarree(),
                      facecolor="none", edgecolor="black", linewidth=4)

# Ajouter les lignes de grille pour la deuxième carte
gl = axs[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=1, linestyle='--')
gl.xlocator = mticker.FixedLocator(
    [120, 140, 160, 180, -160, -140, -120, -100, -80])
gl.ylocator = mticker.FixedLocator([-30, -20, -10, 0, 10, 20, 30])
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 23}
gl.ylabel_style = {'size': 23}

# Ajouter les lignes de côte et les terres pour la deuxième carte
axs[1].coastlines(resolution='50m')
axs[1].add_feature(cfeature.LAND, zorder=100, edgecolor="k")

# Ajouter le titre au deuxième sous-graphiqu
axs[1].set_title("(b) STD of CMIP ΔT' change", fontsize=30, fontweight='bold')
for spine in axs[1].spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)  # plt.ylim([-2, 2])

# Ajuster l'espacement entre les sous-graphiques
plt.tight_layout()
fig.savefig('your_path/figure_1.pdf',
            format='pdf', bbox_inches='tight')
fig.savefig('your_path/figure_1.png',
            format='png', dpi=100, bbox_inches='tight')

# Afficher le graphique
plt.show()


# figure 2
total_coef_array_STD = np.std(total_coef_array, axis=0)
total_coef_array_STD_perc = total_coef_array_STD/total_coef_array_MMM*100

fig, axs = plt.subplots(2, 2, figsize=(40, 20), subplot_kw={
                        'projection': ccrs.PlateCarree(central_longitude=180)})
plt.subplots_adjust(wspace=0.25, hspace=0.1)


def setup_subplot(ax, title, data, cmap='jet', vmin=4, vmax=16):
    ax.add_feature(cfeature.LAND, zorder=100, edgecolor="k")
    ax.set_title(title, fontsize=45, fontweight='bold',
                 pad=20)  # Augmentation taille titre

    cf = ax.contourf(lons_tos_2075_2100[110:300], lats_tos_2075_2100[50:110],
                     data[50:110, 110:300], transform=ccrs.PlateCarree(),
                     cmap=cmap, levels=np.linspace(vmin, vmax, 25))  # Augmentation du nombre de niveaux

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.3, linestyle='--')

    longitudes = np.concatenate([
        np.arange(120, 181, 10),  # De 120°E à 180°E
        np.arange(-190, -71, 10)  # De 180°W à 70°W
    ])
    gl.xlocator = mticker.FixedLocator(longitudes)

    gl.ylocator = mticker.FixedLocator([-30, -20, -10, 0, 10, 20, 30])

    gl.xlabel_style = {'size': 30}  # Taille réduite et rotation ajoutée
    gl.ylabel_style = {'size': 30}
    gl.top_labels = False
    gl.right_labels = False

    ax.coastlines(resolution='50m')
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    return cf


# Créer les subplots avec des paramètres ajustés
cf1 = setup_subplot(axs[0, 0], "(a)   MMM " r"$\mathbf{\alpha}$",
                    total_coef_array_MMM, vmin=2, vmax=16)
cf2 = setup_subplot(axs[0, 1], "(b)   MMM " + r"$\mathbf{\alpha_{LWU}}$",
                    coef_LW_up_array_MMM, vmin=2, vmax=16)
cf3 = setup_subplot(axs[1, 0], "(c)   MMM " + r"$\mathbf{\alpha_{LH}}$",
                    oceanic_latent_coef_array_MMM, vmin=2, vmax=16)
cf4 = setup_subplot(axs[1, 1], "(d)  MMM " + r"$\mathbf{\alpha}$" + " STD",
                    total_coef_array_STD_perc, vmin=0, vmax=10)

# Définir les paramètres des colorbars avec ticks ajustés
cbar_params = [
    {'data': cf1, 'ticks': np.arange(
        2, 16.1, 1), 'units': "W.m$^{-2}$.°C$^{-1}$"},
    {'data': cf2, 'ticks': np.arange(
        2, 16.1, 1), 'units': "W.m$^{-2}$.°C$^{-1}$"},
    {'data': cf3, 'ticks': np.arange(
        0, 16.1, 1), 'units': "W.m$^{-2}$.°C$^{-1}$"},
    {'data': cf4, 'ticks': np.arange(0, 10.1, 1), 'units': "%"}
]

# Ajouter les colorbars
for i, (ax, params) in enumerate(zip(axs.flat, cbar_params)):
    cbar = fig.colorbar(params['data'], ax=ax, orientation='horizontal',
                        ticks=params['ticks'],
                        pad=0.1,
                        aspect=40,
                        shrink=1)

    # Configurer les paramètres de la colorbar
    # Augmentation taille des ticks
    cbar.ax.tick_params(labelsize=40, length=0)
    cbar.set_label(params['units'], size=40, labelpad=10,
                   ha="center")  # Augmentation taille des unités

    # Supprimer les minor ticks
    cbar.ax.minorticks_off()

    # Ajouter les traits noirs verticaux
    for tick in params['ticks']:
        cbar.ax.axvline(x=tick, color='black', linewidth=2, alpha=1, zorder=15)

plt.tight_layout()
fig.savefig('your_path/figure_2.pdf',
            format='pdf', bbox_inches='tight')
fig.savefig('your_path/figure_2.png',
            format='png', dpi=100, bbox_inches='tight')

plt.show()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 18:20:44 2025

@author: vincent
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 18:15:19 2025

@author: vincent
"""


# " figure 3


# Définition des familles de modèles et leurs symboles associés
model_families = {
    'ACCESS': 'o', 'AWI': 's', 'CAMS': '2', 'CAS': 's', 'CCSM': 'v', 'CESM': 'p',
    'CIESM': '*', 'CMCC': 'H', 'CNRM': '+', 'CSIRO': '2', 'CanESM': 'D', 'E3SM': '2',
    'EC-Earth': '3', 'FGOALS': '4', 'FIO': '8', 'GFDL': '<', 'IPSL': '>', 'KIOST': 'P',
    'MIROC': 'd', 'MPI': 'h', 'MRI': 'X', 'NESM': 'p', 'NorESM': '^', 'TaiESM': 's',
    'UKESM': 'P', 'bcc': 'x', 'inmcm': 's'
}

couleurs = ['b', 'g', 'r', 'c', 'm', 'y', 'orange', 'purple', 'pink', 'brown', 'lime', 'teal', 'navy', 'coral', 'olive',
            'gold', 'maroon', 'silver', 'salmon', 'indigo', 'tan', 'khaki', 'crimson', 'orchid', 'turquoise']


def get_model_family(model_name):
    for family in model_families:
        if model_name.startswith(family):
            return family
    return "Other"


modele_couleur_marqueur = {}
color_index = 0

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

for model in model_list_array:
    family = get_model_family(model)

    if family in model_families:
        symbol = model_families[family]
    else:
        symbol = 'o'  # symbole par défaut pour les modèles non catégorisés

    color = couleurs[color_index % len(couleurs)]
    modele_couleur_marqueur[model] = (color, symbol)

    color_index += 1

# Vérification de l'unicité des combinaisons
if len(modele_couleur_marqueur) == len(model_list_array):
    print("Chaque modèle a une combinaison unique de couleur et de symbole.")
else:
    print("Attention : il pourrait y avoir des combinaisons en double.")

# Affichage des assignations pour vérification
for model, (color, symbol) in modele_couleur_marqueur.items():
    print(f"{model}: Couleur = {color}, Symbole = {symbol}")


ax_limit_list = [[-1, 3, -2, 2.5], [-0.1, 2, -0.5, 2], [-2, 0.5, -2.5, 0.5]]

# Définir la grille principale
fig = plt.figure(figsize=(35, 20), facecolor='white')

# Grille pour les deux plots (maps) sur la première ligne
gs_main = GridSpec(2, 2, width_ratios=[
                   1.5, 1.5], wspace=0.15, hspace=0.05, left=0.1, right=0.9)
# Première plot (map) à gauche
ax1 = fig.add_subplot(
    gs_main[0, 0], projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND, zorder=100, edgecolor="k")
ax1.set_title("(a) Model-simulated  MMM ΔT'",
              fontsize=35, fontweight='bold', pad=1)

ccc = 2
step = ccc * 0.05
levels = np.arange(-ccc, ccc + step, step)
cf1 = ax1.contourf(lons_tos_2075_2100[110:300], lats_tos_2075_2100[50:110],
                   relative_tos_change_array_MMM[50:110,
                                                 110:300], transform=ccrs.PlateCarree(),
                   cmap='RdBu_r', levels=levels, extend="both")
gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='black', alpha=0.15, linestyle='--')

ax1.add_geometries([geom_SOUTH], crs=ccrs.PlateCarree(),
                   facecolor="none", edgecolor="black", linewidth=4)
ax1.add_geometries([geom_EAST], crs=ccrs.PlateCarree(),
                   facecolor="none", edgecolor="black", linewidth=4)
ax1.add_geometries([geom_WEST], crs=ccrs.PlateCarree(),
                   facecolor="none", edgecolor="black", linewidth=4)

gl1.xlocator = mticker.FixedLocator(
    [120, 140, 160, 180, -160, -140, -120, -100, -80])
gl1.ylocator = mticker.FixedLocator([-30, -20, -10, 0, 10, 20, 30])
gl1.xlabel_style = {'size': 25, 'color': 'black'}
gl1.ylabel_style = {'size': 30, 'color': 'black'}
gl1.top_labels = False
gl1.right_labels = False
ax1.coastlines(resolution='50m')
ax1.set_frame_on(True)
for spine in ax1.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# Deuxième plot (map) à droite
ax2 = fig.add_subplot(
    gs_main[0, 1], projection=ccrs.PlateCarree(central_longitude=180))
ax2.add_feature(cfeature.LAND, zorder=100, edgecolor="k")
ax2.set_title("(b) Equation-derived MMM ΔT'", fontsize=35, fontweight='bold')
cf2 = ax2.contourf(lons_tos_2075_2100[110:300], lats_tos_2075_2100[50:110],
                   RSST_somme_component_change_ALPHA_MMM_MMM[50:110, 110:300], transform=ccrs.PlateCarree(
),
    cmap='RdBu_r', levels=levels, extend="both")
gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='black', alpha=0.15, linestyle='--')

ax2.add_geometries([geom_SOUTH], crs=ccrs.PlateCarree(),
                   facecolor="none", edgecolor="black", linewidth=4)
ax2.add_geometries([geom_EAST], crs=ccrs.PlateCarree(),
                   facecolor="none", edgecolor="black", linewidth=4)
ax2.add_geometries([geom_WEST], crs=ccrs.PlateCarree(),
                   facecolor="none", edgecolor="black", linewidth=4)
gl2.xlocator = mticker.FixedLocator(
    [120, 140, 160, 180, -160, -140, -120, -100, -80])
gl2.ylocator = mticker.FixedLocator([-30, -20, -10, 0, 10, 20, 30])
gl2.xlabel_style = {'size': 25, 'color': 'black'}
gl2.ylabel_style = {'size': 30, 'color': 'black'}
gl2.top_labels = False
gl2.left_labels = False
gl2.right_labels = False
ax2.coastlines(resolution='50m')
ax2.set_frame_on(True)
for spine in ax2.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# Ajouter la barre de couleur partagée à droite
# [left, bottom, width, height]
cbar_ax = fig.add_axes([0.92, 0.56, 0.0075, 0.25])
cbar = fig.colorbar(cf1, cax=cbar_ax, orientation="vertical", label="°C")

cbar.ax.tick_params(labelsize=20)  # Taille des indices de la colorbar
# Taille et gras pour le label de la colorbar
cbar.set_label("°C", fontsize=25, fontweight='bold')

# Grille pour les trois derniers plots (scatterplots) sur la deuxième ligne
gs_scatter = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[1, :])

# Données
data_list = [
    (relative_tos_CMIPS_WEST_change_mean, RSST_somme_component_CMIPS_WEST_change_mean, "(c) WEP",
     "Model-simulated ΔT' (°C)", "Equation-derived ΔT' (°C)", ax_limit_list[0]),
    (relative_tos_CMIPS_EAST_change_mean, RSST_somme_component_CMIPS_EAST_change_mean, "(d) EEP",
     "Model-simulated ΔT' (°C)", "Equation-derived ΔT' (°C)", ax_limit_list[1]),
    (relative_tos_CMIPS_SOUTH_change_mean, RSST_somme_component_CMIPS_SOUTH_change_mean, "(e) SEP",
     "Model-simulated ΔT' (°C)", "Equation-derived ΔT' (°C)", ax_limit_list[2]),
]


def plot_data(ax, x, y, model_list, title, xlabel, ylabel, ax_limit):
    stat, p = shapiro(y)
    alpha = 0.05

    slope, intercept, r_value, pv, se = stats.linregress(x, y)
    rho, pval = stats.spearmanr(x, y)
    sns.regplot(x=x, y=y, color="black", ci=None, scatter=False,
                marker='+', scatter_kws={"s": 100}, ax=ax)

    for i in range(len(model_list)):
        modele = model_list[i]
        couleur, marqueur = modele_couleur_marqueur[modele]
        ax.scatter(x[i], y[i], label=modele,
                   marker=marqueur, color=couleur, s=500)

    ax.axhline(0, linestyle="--", color="black")
    ax.axvline(0, linestyle="--", color="black")

    if p > alpha:
        print('Sample2 looks Gaussian (fail to reject H0)')
        legend_text = 'slope=%.3f, r=%.3f' % (slope, r_value)
    else:
        print('Sample2 does not look Gaussian (reject H0)')
        legend_text = 'slope=%.3f, r=%.3f' % (slope, rho)

    # Ajout du texte de la légende directement sur le graphique avec un cadre
    ax.text(0.05, 0.95, legend_text, transform=ax.transAxes, fontsize=30, weight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black', alpha=0.7, lw=1))

    ax.set_xlabel(xlabel, fontsize=35)
    ax.set_ylabel(ylabel, fontsize=35)
    ax.tick_params(axis='both', labelsize=25)
    ax.axis(ax_limit)

    # Ajout d'un espacement pour le titre
    ax.set_title(title, fontsize=35, fontweight='bold', pad=15)

    # Encadrement du subplot
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    # Fond blanc pour le subplot
    ax.set_facecolor('white')

    # Ajout d'une grille
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray')


# Boucle pour les trois scatterplots
for i in range(3):
    x, y, title, xlabel, ylabel, ax_limit = data_list[i]
    ax = fig.add_subplot(gs_scatter[0, i])
    plot_data(ax, x, y, model_list, title, xlabel, ylabel, ax_limit)

# Ajustement de la mise en page
plt.tight_layout()
legend_elements = []
for model in model_list_array:
    couleur, marker = modele_couleur_marqueur[model]
    legend_elements.append(Line2D([0], [0], marker=marker, color=couleur, markerfacecolor=couleur,
                                  linestyle='None', markersize=10, label=model))

# Tri des éléments de la légende par famille de modèles
legend_elements.sort(key=lambda x: get_model_family(x.get_label()))

legend = plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(-0.65, -0.8),
                    ncol=8, fontsize=25, markerscale=2.5, frameon=True, fancybox=True, shadow=True)

# Ajout d'un cadre à la légende
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
frame.set_linewidth(2)
fig.savefig('your_path/figure_3.pdf',
            format='pdf', bbox_inches='tight')
fig.savefig('your_path/figure_3.png',
            format='png', dpi=100, bbox_inches='tight')

# Affichage de la figure
plt.show()


######################################################## figure 4 ###########################


pas = 6
tauu_pas = tauu_change_array_norm[:, 50:110,
                                  110:300][:, ::pas, ::pas]  # (9,28)
tauu_pas_MMM = np.nanmean(tauu_pas, axis=0)
tauv_pas = tauv_change_array_norm[:, 50:110,
                                  110:300][:, ::pas, ::pas]  # (9,28)
tauv_pas_MMM = np.nanmean(tauv_pas, axis=0)


uas_change_array_median = np.nanmean(uas_change_array, axis=0)
vas_change_array_median = np.nanmean(vas_change_array, axis=0)

uas_change_array_norm = uas_change_array / \
    np.sqrt(uas_change_array**2+vas_change_array**2)
uas_change_array_norm_median = np.nanmean(uas_change_array_norm, axis=0)
vas_change_array_norm = vas_change_array / \
    np.sqrt(uas_change_array**2+vas_change_array**2)
vas_change_array_norm_median = np.nanmean(vas_change_array_norm, axis=0)

uas_pas = uas_change_array_norm_median[50:110, 110:300][::pas, ::pas]  # (9,28)
vas_pas = vas_change_array_norm_median[50:110, 110:300][::pas, ::pas]  # (9,28)

clt_change_array_MMM = np.nanmean(clt_change_array, axis=0)
clt_change_array_STD = np.std(clt_change_array, axis=0)


lon_pas = lon_Pacific[::pas]  # 28
lat_pas = lat_Pacific[::pas]  # 9
# Create a grid of subplots (4 rows, 2 columns)

# Modification de la disposition: 4 sous-figures dans la première colonne, 4 dans la deuxième
fig, axs = plt.subplots(4, 2, figsize=(25, 20), subplot_kw={
                        'projection': ccrs.PlateCarree(central_longitude=180)})
fig.subplots_adjust(hspace=0.3, wspace=0.1)

# Define common parameters
ccc = 2
step = ccc * 0.05
levels = np.arange(-ccc, ccc + step, step)

# Nouvel ordre des sous-figures selon votre demande
# Première colonne: ΔT', ΔQ'For, ΔO'*, ΔTα'*
# Deuxième colonne: ΔLWD'*, ΔLH'For*, ΔSW'*, ΔSH'*
index_order = [
    0,  # (a) ΔT'
    3,  # (b) ΔQ'For
    1,  # (c) ΔO'*
    2,  # (d) ΔTα'*
    4,  # (e) ΔLWD'*
    5,  # (f) ΔLH'For*
    6,  # (g) ΔSW'*
    7   # (h) ΔSH'*
]

# Liste des titres pour chaque sous-figure
titles = [
    r"(a) MMM $\mathbf{\Delta T}'$",
    r"(b) MMM $\mathbf{\Delta Q_{For}'*}$",
    r"(c) MMM $\mathbf{\Delta O'*}$",
    r"(d) MMM $\mathbf{\overline{\Delta T}\alpha^\prime*}$",
    r"(e) MMM $\mathbf{\Delta LWD'*}$",
    r"(f) MMM $\mathbf{\Delta LH_{For}'*}$",
    r"(g) MMM $\mathbf{\Delta SW'*}$",
    r"(h) MMM $\mathbf{\Delta SH'*}$"
]

# Ajouter des annotations pour les relations entre les figures
annotations = [
    "(= b + c + d)",   # Pour (a) MMM ΔT'
    "(= e + g + f + h)",  # Pour (b) MMM ΔQ'For
    "",                # Pour (c) MMM ΔO'*
    "",                # Pour (d) MMM ΔTα'*
    "",                # Pour (e) MMM ΔLWD'*
    "",                # Pour (f) MMM ΔLH'For*
    "",                # Pour (g) MMM ΔSW'*
    ""                 # Pour (h) MMM ΔSH'*
]

# Réorganisation des variables selon le nouvel ordre
var_plot_list_initial = np.array([
    RSST_somme_component_change_ALPHA_MMM_MMM,           # (a) ΔT'
    # (anciennement (b), maintenant (c)) ΔO'*
    Patterns_relative_D0_change_array_MMM,
    # (anciennement (c), maintenant (d)) ΔTα'*
    oceanic_flux_deviation_array_fraction_MMM,
    # (anciennement (d), maintenant (b)) ΔQ'For
    Pattern_atm_somme_component_change_MMM,
    Patterns_atm_variable_total_array_MMM[2],            # (e) ΔLWD'*
    Patterns_atm_variable_total_array_MMM[3],            # (f) ΔLH'For*
    Patterns_atm_variable_total_array_MMM[0],            # (g) ΔSW'*
    Patterns_atm_variable_total_array_MMM[1]             # (h) ΔSH'*
])

# Réorganiser le tableau des variables selon le nouvel ordre
var_plot_list = var_plot_list_initial[index_order]

# Function to set up common elements for each subplot


def setup_subplot(ax, title, annotation, row, col):
    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.LAND, zorder=100, edgecolor="k")
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.2, linestyle='--')
    gl.xlocator = mticker.FixedLocator(
        [120, 140, 160, 180, -160, -140, -120, -100, -80])
    gl.ylocator = mticker.FixedLocator([-30, -20, -10, 0, 10, 20, 30])
    gl.xlabel_style = {'size': 20, 'color': 'black'}
    gl.ylabel_style = {'size': 20, 'color': 'black'}
    gl.top_labels = False
    gl.right_labels = False

    if col == 1:
        gl.left_labels = False

    if row != 3:  # Not the last row
        gl.bottom_labels = False

    # Titre principal
    if annotation:
        # Si une annotation existe, combiner le titre et l'annotation
        combined_title = f"{title} {annotation}"
        # Taille réduite pour les titres avec annotations
        ax.set_title(combined_title, fontsize=28, fontweight='bold', pad=20)
    else:
        # Sinon, utiliser uniquement le titre
        ax.set_title(title, fontsize=28, fontweight='bold', pad=20)


# Correspondance entre l'indice et la position dans la figure (numérotation classique)
positions = [
    (0, 0),  # a (1ère ligne, 1ère colonne)
    (1, 0),  # b (2ème ligne, 1ère colonne)
    (2, 0),  # c (3ème ligne, 1ère colonne)
    (3, 0),  # d (4ème ligne, 1ère colonne)
    (0, 1),  # e (1ère ligne, 2ème colonne)
    (1, 1),  # f (2ème ligne, 2ème colonne)
    (2, 1),  # g (3ème ligne, 2ème colonne)
    (3, 1)  # h (4ème ligne, 2ème colonne)
]

# Tableau pour suivre les indices originaux des sous-figures réordonnées
subplot_indices = {}

# Plotting each subplot with the new order
for i, (row, col) in enumerate(positions):
    ax = axs[row, col]
    setup_subplot(ax, titles[i], annotations[i], row, col)

    # Stocker l'index original de la sous-figure pour les conditions spéciales
    orig_idx = index_order[i]
    subplot_indices[i] = orig_idx

    cf = ax.contourf(lon_Pacific, lat_Pacific, var_plot_list[i, 50:110, 110:300],
                     transform=ccrs.PlateCarree(), cmap='RdBu_r', levels=levels, extend="both")

    # Conditions spéciales basées sur l'indice original (oceanic_flux_deviation_array_fraction_MMM)
    if orig_idx == 2:  # ΔTα'* (maintenant en position (d))
        contour = ax.contour(lon_Pacific, lat_Pacific, relative_total_coef_array_MMM[50:110, 110:300],
                             transform=ccrs.PlateCarree(), colors='black',
                             levels=[-5, -2.5, 0, 2.5, 5], alpha=0.8)
        # Pour chaque niveau de contour
        for level in contour.levels:
            # Trouver les coordonnées des lignes de contour pour ce niveau
            for path in contour.collections[int((level + 5)/2.5)].get_paths():
                # Prendre le point médian du contour
                vertices = path.vertices
                mid_point = vertices[len(vertices)//2]
                # Ajouter le texte à ce point
                ax.text(mid_point[0], mid_point[1], f'{level:.1f}',
                        fontsize=18, color='black', transform=ccrs.PlateCarree())

    # Conditions spéciales basées sur l'indice original (Patterns_atm_variable_total_array_MMM[0])
    if orig_idx == 6:  # ΔSW'* (toujours en position (g))
        contour = ax.contour(lon_Pacific, lat_Pacific, clt_change_array_MMM[50:110, 110:300],
                             transform=ccrs.PlateCarree(), colors='black', levels=[-5, -2.5, 0, 2.5, 5], alpha=0.8)
        ax.clabel(contour, inline=True, fontsize=18, colors='black')

    # Conditions spéciales basées sur l'indice original (Patterns_relative_D0_change_array_MMM)
    if orig_idx == 1:  # ΔO'* (maintenant en position (c))
        qv = ax.quiver(lon_pas, lat_pas, tauu_pas_MMM, tauv_pas_MMM, pivot='middle', alpha=0.5,
                       transform=ccrs.PlateCarree())
        ax.quiverkey(qv, X=0.8, Y=-0.14, U=1, label='1 N.m⁻²', labelpos='E', coordinates='axes',
                     fontproperties={'size': 25})

    # Conditions spéciales basées sur l'indice original (Patterns_atm_variable_total_array_MMM[3])
    if orig_idx == 5:  # ΔLH'For* (toujours en position (f))
        qv = ax.quiver(lon_pas, lat_pas, uas_pas, vas_pas, pivot='middle', alpha=0.5,
                       transform=ccrs.PlateCarree())
        ax.quiverkey(qv, X=0.8, Y=-0.14, U=1, label='1 m.s⁻¹', labelpos='E', coordinates='axes',
                     fontproperties={'size': 25})

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)


# Add colorbar
cax = fig.add_axes([0.1, 0.075, 0.8, 0.02])
cbar = plt.colorbar(cf, cax=cax, orientation="horizontal")
cbar.set_label(label='°C', fontsize=30, fontweight='bold')
cbar.ax.tick_params(labelsize=30)

# Sauvegarder en format PDF
fig.savefig('your_path/figure_4.pdf',
            format='pdf', bbox_inches='tight')

# Sauvegarder en format PNG avec une résolution de 300 dpi
fig.savefig('your_path/figure_4.png',
            format='png', dpi=100, bbox_inches='tight')

plt.show()



####################### figure 5 ############################


def convert_to_percentages(data):
    rsst_variance = data['RSST']
    return {key: (value / rsst_variance) * 100 for key, value in data.items() if key != 'RSST'}


# Liste des noms de colonnes mise à jour avec du texte en gras
name_DF_list = [
    r"$\mathbf{\Delta T'}$",
    r"$\mathbf{\Delta O'*}$",
    r"$\mathbf{\overline{\Delta T}\alpha'*}$",
    r"$\mathbf{\Delta Q_{For}'*}$",
    r"$\mathbf{\Delta LWD'*}$",
    r"$\mathbf{\Delta LH_{For}'*}$",
    r"$\mathbf{\Delta SW'*}$",
    r"$\mathbf{\Delta SH'*}$"
]

# Définition des couleurs
# 4 premières couleurs différentes
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
base_color = '#d62728'  # Couleur du 4ème barplot (ΔFor')
n_shades = 3  # Réduit à 3 au lieu de 4
color_shades = [plt.cm.Reds(i/n_shades) for i in range(1, n_shades+1)]
# Ajout d'une couleur marron pour l'avant-dernière barre
all_colors = colors + color_shades + ['#8B4513']

# Combinaison des couleurs
all_colors = colors + color_shades

# Configuration de la figure
fig = plt.figure(figsize=(40, 35), facecolor='white')
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1])
fig.subplots_adjust(top=0.9, hspace=0.3, wspace=0.3, bottom=0.1)


def create_subplot(gs_pos, data, title, ax_limit=(-2.75, 2.75)):
    ax = plt.subplot(gs[gs_pos])
    sns.boxplot(data=data, ax=ax, palette=all_colors)
    ax.axhline(y=0, color='black', linewidth=2, linestyle='--')
    ax.vlines(x=[3.5], ymin=ax_limit[0], ymax=ax_limit[1],
              color='red', linestyle='solid', linewidth=5)
    ax.set_ylim(ax_limit)
    ax.tick_params(axis='y', labelsize=40)
    ax.set_ylabel("°C", fontsize=45)

    # Modification des labels x
    ax.set_xticklabels(name_DF_list, fontsize=45,
                       fontweight='bold', rotation=40, ha='center')

    ax.set_title(title, fontsize=50, fontweight='bold', pad=20)
    ax.set_facecolor('white')
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
    ax.set_yticks(np.arange(ax_limit[0], ax_limit[1]+0.5, 0.5))
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)


# Création des subplots
create_subplot((0, 0), Df_mean_components_CMIPS_PATTERNS_SOUTH, "(a) SEP")
create_subplot((0, 1), Df_mean_components_CMIPS_PATTERNS_WEST, "(b) WEP")
create_subplot((1, 0), Df_mean_components_CMIPS_PATTERNS_EAST, "(c) EEP")
create_subplot(
    (1, 1), Df_mean_components_CMIPS_PATTERNS_GRADIENT, "(d) EEP - WEP")

# Ajustement des marges
plt.subplots_adjust(left=0.05, right=0.95, top=0.95,
                    bottom=0.15, wspace=0.25, hspace=0.3)
fig.savefig('your_path/figure_5.pdf',
            format='pdf', bbox_inches='tight')
fig.savefig('your_path/figure_5.png',
            format='png', dpi=100, bbox_inches='tight')

plt.show()


# Configuration des couleurs et noms
colors = ['#ff7f0e', '#2ca02c', '#d62728']
base_color = '#d62728'
n_shades = 3
color_shades = [plt.cm.Reds(i/n_shades) for i in range(1, n_shades+1)]
all_colors = colors + color_shades + ['#8B4513']


# " figure 7


# Définition des familles de modèles et leurs symboles associés
model_families = {
    'ACCESS': 'o', 'AWI': 's', 'CAMS': '2', 'CAS': 's', 'CCSM': 'v', 'CESM': 'p',
    'CIESM': '*', 'CMCC': 'H', 'CNRM': '+', 'CSIRO': '2', 'CanESM': 'D', 'E3SM': '2',
    'EC-Earth': '3', 'FGOALS': '4', 'FIO': '8', 'GFDL': '<', 'IPSL': '>', 'KIOST': 'P',
    'MIROC': 'd', 'MPI': 'h', 'MRI': 'X', 'NESM': 'p', 'NorESM': '^', 'TaiESM': 's',
    'UKESM': 'P', 'bcc': 'x', 'inmcm': 's'
}

couleurs = ['b', 'g', 'r', 'c', 'm', 'y', 'orange', 'purple', 'pink', 'brown', 'lime', 'teal', 'navy', 'coral', 'olive',
            'gold', 'maroon', 'silver', 'salmon', 'indigo', 'tan', 'khaki', 'crimson', 'orchid', 'turquoise']


def get_model_family(model_name):
    for family in model_families:
        if model_name.startswith(family):
            return family
    return "Other"


modele_couleur_marqueur = {}
color_index = 0

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

for model in model_list_array:
    family = get_model_family(model)

    if family in model_families:
        symbol = model_families[family]
    else:
        symbol = 'o'  # symbole par défaut pour les modèles non catégorisés

    color = couleurs[color_index % len(couleurs)]
    modele_couleur_marqueur[model] = (color, symbol)

    color_index += 1

# Vérification de l'unicité des combinaisons
if len(modele_couleur_marqueur) == len(model_list_array):
    print("Chaque modèle a une combinaison unique de couleur et de symbole.")
else:
    print("Attention : il pourrait y avoir des combinaisons en double.")

# Affichage des assignations pour vérification
for model, (color, symbol) in modele_couleur_marqueur.items():
    print(f"{model}: Couleur = {color}, Symbole = {symbol}")


def calculate_common_limits(data_lists):
    all_data = np.concatenate(data_lists)
    min_val, max_val = np.min(all_data), np.max(all_data)
    margin = (max_val - min_val) * 0.1  # 10% margin
    return min_val - margin, max_val + margin


def create_subplot_percentage(ax, df_melted, title):
    sns.barplot(x='Variable', y='Percentage',
                data=df_melted, ax=ax, palette=all_colors)

    ax.set_ylim(global_y_min, global_y_max)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax.vlines(x=[2.5], ymin=global_y_min, ymax=global_y_max,
              color='red', linestyle='solid', linewidth=5)

    for i, p in enumerate(ax.patches):
        height = p.get_height()
        if np.isnan(height):
            continue
        y_pos = height + 0.02 * \
            (global_y_max - global_y_min) if height > 0 else height - \
            0.02*(global_y_max - global_y_min)
        va = 'bottom' if height > 0 else 'top'
        ax.text(p.get_x() + p.get_width() / 2., y_pos, f'{height:.1f}%',
                ha='center', va=va, color='black', fontweight='bold', fontsize=30)

    ax.tick_params(axis='y', labelsize=40)
    ax.set_ylabel("%", fontsize=40)
    ax.set_xticklabels(name_DF_list, fontsize=40,
                       fontweight='bold', rotation=40, ha='center')
    ax.set_title(title, fontsize=50, fontweight='bold', pad=20)
    ax.set_facecolor('white')
    ax.set_xlabel('')
    ax.grid(True, linestyle='--', linewidth=1, color='gray')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(3)


def create_scatter_plot(ax, x, y, title, model_list_array, modele_couleur_marqueur, xlabel, legend_pos, xlim, ylim):
    stat, p = stats.shapiro(y)
    slope, intercept, r_value, pv, se = stats.linregress(x, y)
    rho, pval = stats.spearmanr(x, y)

    sns.regplot(x=x, y=y, color="black", ci=None,
                scatter=False, ax=ax, line_kws={'linewidth': 3})

    for i, modele in enumerate(model_list_array):
        couleur, marqueur = modele_couleur_marqueur[modele]
        ax.scatter(x[i], y[i], label=modele,
                   marker=marqueur, color=couleur, s=1000)

    ax.axhline(0, linestyle="--", color="black", linewidth=2)
    ax.axvline(0, linestyle="--", color="black", linewidth=2)

    legend_text = f'r={r_value:.3f}'
    weight = 'bold' if pv < 0.05 else 'normal'
    ax.annotate(legend_text, xy=legend_pos, xycoords='axes fraction',
                fontsize=43, va='top', ha='left', weight=weight,
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black', alpha=0.7, lw=2))

    ax.set_xlabel(xlabel, fontsize=50)
    ax.set_ylabel(" ΔT' (°C)", fontsize=50)
    ax.set_facecolor('white')
    ax.tick_params(axis='both', labelsize=35)
    ax.grid(True, linestyle='--', linewidth=1, color='gray')
    ax.set_title(title, fontsize=45, fontweight='bold', pad=20)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(3)


# Préparation des données pour les barplots
data_WEST_percentage = convert_to_percentages(data_WEST)
data_EAST_percentage = convert_to_percentages(data_EAST)
data_GRADIENT_percentage = convert_to_percentages(data_GRADIENT)

df_rsst_WEST_percentage = pd.DataFrame(data_WEST_percentage, index=[0])
df_rsst_EAST_percentage = pd.DataFrame(data_EAST_percentage, index=[0])
df_rsst_GRADIENT_percentage = pd.DataFrame(data_GRADIENT_percentage, index=[0])

df_melted_WEST_percentage = df_rsst_WEST_percentage.melt(
    var_name='Variable', value_name='Percentage')
df_melted_EAST_percentage = df_rsst_EAST_percentage.melt(
    var_name='Variable', value_name='Percentage')
df_melted_GRADIENT_percentage = df_rsst_GRADIENT_percentage.melt(
    var_name='Variable', value_name='Percentage')

global_y_min = -120
global_y_max = 200

# Préparation des données pour les scatter plots
x_data_1 = [Patterns_relative_D0_CMIPS_WEST_change_mean, Patterns_relative_D0_CMIPS_EAST_change_mean,
            Patterns_relative_D0_CMIPS_EAST_change_mean-Patterns_relative_D0_CMIPS_WEST_change_mean]
y_data = [relative_tos_CMIPS_WEST_change_mean, relative_tos_CMIPS_EAST_change_mean,
          relative_tos_CMIPS_EAST_change_mean-relative_tos_CMIPS_WEST_change_mean]

xlim_1 = calculate_common_limits(x_data_1)
ylim = calculate_common_limits(y_data)

# Création de la figure principale
fig = plt.figure(figsize=(45, 25))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

# Première ligne : barplots
create_subplot_percentage(plt.subplot(
    gs[0, 0]), df_melted_WEST_percentage, "(a) WEP")
create_subplot_percentage(plt.subplot(
    gs[0, 1]), df_melted_EAST_percentage, "(b) EEP")
create_subplot_percentage(plt.subplot(
    gs[0, 2]), df_melted_GRADIENT_percentage, "(c) EEP - WEP")

# Deuxième ligne : scatter plots
legend_positions = [(0.05, 0.95), (0.75, 0.1), (0.75, 0.1)]

create_scatter_plot(plt.subplot(gs[1, 0]), Patterns_relative_D0_CMIPS_WEST_change_mean, relative_tos_CMIPS_WEST_change_mean,
                    "(d) ΔT' vs ΔO'* WEP", model_list_array, modele_couleur_marqueur, "ΔO'* (°C)", legend_positions[0], xlim_1, ylim)
create_scatter_plot(plt.subplot(gs[1, 1]), Patterns_relative_D0_CMIPS_EAST_change_mean, relative_tos_CMIPS_EAST_change_mean,
                    "(e) ΔT' vs ΔO'* EEP", model_list_array, modele_couleur_marqueur, "ΔO'* (°C)", legend_positions[1], xlim_1, ylim)
create_scatter_plot(plt.subplot(gs[1, 2]), Patterns_relative_D0_CMIPS_EAST_change_mean-Patterns_relative_D0_CMIPS_WEST_change_mean,
                    relative_tos_CMIPS_EAST_change_mean-relative_tos_CMIPS_WEST_change_mean,
                    "(f) ΔT' vs ΔO'* EEP-WEP", model_list_array, modele_couleur_marqueur, "ΔO'*(°C)", legend_positions[2], xlim_1, ylim)

plt.subplots_adjust(left=0.05, right=0.95, top=0.95,
                    bottom=0.05, wspace=0.2, hspace=0.40)
fig.savefig('your_path/figure_7.pdf',
            format='pdf', bbox_inches='tight')
fig.savefig('your_path/figure_7.png',
            format='png', dpi=100, bbox_inches='tight')

plt.show()



#############################################" figure 8 ###########################


tauu_change_array_MMM = np.nanmean(tauu_change_array, axis=0)
tauu_change_array_STD = np.std(tauu_change_array, axis=0)


tos_biais_array_MMM = np.nanmean(tos_biais_array, axis=0)


def create_scatter_plot(ax, x, y, title, model_list_array, modele_couleur_marqueur, xlabel, legend_pos, xlim, ylim):
    stat, p = stats.shapiro(y)
    slope, intercept, r_value, pv, se = stats.linregress(x, y)
    rho, pval = stats.spearmanr(x, y)

    sns.regplot(x=x, y=y, color="black", ci=None, scatter=False, ax=ax)

    for i, modele in enumerate(model_list_array):
        couleur, marqueur = modele_couleur_marqueur[modele]
        ax.scatter(x[i], y[i], label=modele,
                   marker=marqueur, color=couleur, s=1000)

    ax.axhline(0, linestyle="--", color="black")
    ax.axvline(0, linestyle="--", color="black")

    legend_text = f'r={r_value:.3f}'
    weight = 'bold' if pv < 0.05 else 'normal'
    ax.annotate(legend_text, xy=legend_pos, xycoords='axes fraction',
                fontsize=55, va='top', ha='left', weight=weight,
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black', alpha=0.7, lw=1))

    ax.set_xlabel(xlabel, fontsize=65)
    ax.set_ylabel(" ΔO'(°C) ", fontsize=65)
    ax.set_facecolor('white')
    ax.tick_params(axis='both', labelsize=50)
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
    ax.set_title(title, fontsize=65, fontweight='bold', pad=50)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    return ax


# Create figure with increased size
fig = plt.figure(figsize=(70, 30))

# Modified GridSpec for better spacing
gs = gridspec.GridSpec(2, 3,
                       width_ratios=[1.5, 1, 1],
                       height_ratios=[1, 1],
                       wspace=0.35,
                       hspace=0.35,
                       left=0.05,
                       right=0.98,
                       top=0.95,
                       bottom=0.08)

# First line, first column: Wind stress map
ax1 = fig.add_subplot(
    gs[0, 0], projection=ccrs.PlateCarree(central_longitude=180))

ccc2 = 25  # Converted to 10⁻³ N.m⁻²
step2 = ccc2 * 0.05
levels2 = np.arange(-ccc2, ccc2 + step2, step2)

contourf2 = ax1.contourf(lons_tos_2075_2100[110:300], lats_tos_2075_2100[50:110],
                         tauu_change_array_MMM[50:110, 110:300] * 1000,
                         transform=ccrs.PlateCarree(), cmap='RdBu_r', levels=levels2, extend="both")

cbar2 = fig.colorbar(contourf2, ax=ax1, orientation="horizontal", label="N.m⁻²",
                     fraction=0.046, pad=0.1, aspect=45)
cbar2.ax.tick_params(labelsize=55)
cbar2.set_label("10⁻³ N.m⁻²", size=60, labelpad=15)

qv = ax1.quiver(lon_pas, lat_pas, tauu_pas_MMM, tauv_pas_MMM, pivot='middle', alpha=0.5,
                transform=ccrs.PlateCarree())
ax1.quiverkey(qv, X=1, Y=-0.12, U=1, label='1 N.m⁻²', labelpos='E', coordinates='axes',
              fontproperties={'size': 60})
ax1.add_geometries([geom_EAST], crs=ccrs.PlateCarree(),
                   facecolor="none", edgecolor="black", linewidth=4)
ax1.add_geometries([geom_WEST], crs=ccrs.PlateCarree(),
                   facecolor="none", edgecolor="black", linewidth=4)
gl2 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=1, linestyle='--')
gl2.xlocator = mticker.FixedLocator(
    [120, 140, 160, 180, -160, -140, -120, -100, -80])
gl2.ylocator = mticker.FixedLocator([-30, -20, -10, 0, 10, 20, 30])
gl2.top_labels = False
gl2.right_labels = False
gl2.xlabel_style = {'size': 45}
gl2.ylabel_style = {'size': 45}

ax1.coastlines(resolution='50m')
ax1.add_feature(cfeature.LAND, zorder=100, edgecolor="k")
ax1.set_title("(a)  MMM Δu'", fontsize=65, fontweight='bold', pad=30)

# First line, second and third columns: Wind stress scatter plots
ylim = (-3, 3)
xlim_2 = (-12, 21)  # Converted to 10⁻³ N.m⁻²

ax_scatter1 = create_scatter_plot(fig.add_subplot(gs[0, 1]),
                                  relative_tauu_CMIPS_WEST_change_mean*1000,
                                  Patterns_relative_D0_CMIPS_WEST_change_mean,
                                  "(b) Δu' vs ΔO'* in WEP",
                                  model_list_array, modele_couleur_marqueur,
                                  "Δu' (10⁻³ N.m⁻²)",
                                  (0.05, 0.95), xlim_2, ylim)

ax_scatter1.set_ylabel("ΔO'*(°C)", fontsize=65)

ax_scatter2 = create_scatter_plot(fig.add_subplot(gs[0, 2]),
                                  relative_tauu_CMIPS_EAST_change_mean*1000,
                                  Patterns_relative_D0_CMIPS_EAST_change_mean,
                                  "(c) Δu' vs ΔO'* in EEP '",
                                  model_list_array, modele_couleur_marqueur,
                                  "Δu' (10⁻³ N.m⁻²)",
                                  (0.05, 0.95), xlim_2, ylim)

ax_scatter2.set_ylabel("ΔO'*(°C)", fontsize=65)

# Second line, first column: Temperature bias map
ax3 = fig.add_subplot(
    gs[1, 0], projection=ccrs.PlateCarree(central_longitude=180))

ccc1 = 3
step1 = ccc1 * 0.05
levels1 = np.arange(-ccc1, ccc1 + step1, step1)

contourf1 = ax3.contourf(lons_tos_2075_2100[110:300], lats_tos_2075_2100[50:110],
                         tos_biais_array_MMM[50:110, 110:300],
                         transform=ccrs.PlateCarree(), cmap='RdBu_r', levels=levels1, extend="both")

cbar1 = fig.colorbar(contourf1, ax=ax3, orientation="horizontal", label="°C",
                     fraction=0.046, pad=0.1, aspect=45)
cbar1.ax.tick_params(labelsize=55)
cbar1.set_label("°C", size=60, labelpad=15)

ax3.add_geometries([geom_EAST], crs=ccrs.PlateCarree(),
                   facecolor="none", edgecolor="black", linewidth=4)
ax3.add_geometries([geom_WEST], crs=ccrs.PlateCarree(),
                   facecolor="none", edgecolor="black", linewidth=4)

gl1 = ax3.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=1, linestyle='--')
gl1.xlocator = mticker.FixedLocator(
    [120, 140, 160, 180, -160, -140, -120, -100, -80])
gl1.ylocator = mticker.FixedLocator([-30, -20, -10, 0, 10, 20, 30])
gl1.top_labels = False
gl1.right_labels = False
gl1.xlabel_style = {'size': 45}
gl1.ylabel_style = {'size': 45}

ax3.coastlines(resolution='50m')
ax3.add_feature(cfeature.LAND, zorder=100, edgecolor="k")
ax3.set_title("(d)  MMM T bias", fontsize=65, fontweight='bold', pad=40)

# Second line, second and third columns: Temperature bias scatter plots
xlim_1 = (-4, 1)

ax_scatter3 = create_scatter_plot(fig.add_subplot(gs[1, 1]),
                                  tos_biais_CMIPS_WEST_change_mean,
                                  Patterns_relative_D0_CMIPS_WEST_change_mean,
                                  "(e) T bias vs ΔO'* in WEP ",
                                  model_list_array, modele_couleur_marqueur,
                                  "T bias (°C) ",
                                  (0.75, 0.1), xlim_1, ylim)

ax_scatter3.set_ylabel("ΔO'*(°C)", fontsize=65)

ax_scatter4 = create_scatter_plot(fig.add_subplot(gs[1, 2]),
                                  tos_biais_CMIPS_EAST_change_mean,
                                  Patterns_relative_D0_CMIPS_EAST_change_mean,
                                  "(f) T bias vs ΔO'* in EEP ",
                                  model_list_array, modele_couleur_marqueur,
                                  "T bias (°C) ",
                                  (0.75, 0.1), xlim_1, ylim)

ax_scatter4.set_ylabel("ΔO'*(°C)", fontsize=65)

# Adjust borders for all maps
for ax in [ax1, ax3]:
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

fig.savefig('your_path/figure_8.pdf',
            format='pdf', bbox_inches='tight')
fig.savefig('your_path/figure_8.png',
            format='png', dpi=100, bbox_inches='tight')


plt.show()


#################################################################################### figure S4
# Figure 8: Wind Curl, Wind Bias and their relationship with oceanic dynamics
fig = plt.figure(figsize=(70, 30))

# Modified GridSpec for better spacing
gs = gridspec.GridSpec(2, 3,
                       width_ratios=[1.5, 1, 1],
                       height_ratios=[1, 1],
                       wspace=0.35,
                       hspace=0.35,
                       left=0.05,
                       right=0.98,
                       top=0.95,
                       bottom=0.08)

# First line, first column: Wind curl map
ax1 = fig.add_subplot(
    gs[0, 0], projection=ccrs.PlateCarree(central_longitude=180))

# Ajustez ces valeurs selon la plage de votre curl du vent
ccc2 = 0.4  # en 10⁻⁷ N.m⁻³
step2 = ccc2 * 0.05
levels2 = np.arange(-ccc2, ccc2 + step2, step2)

contourf2 = ax1.contourf(lons_tos_2075_2100[110:300], lats_tos_2075_2100[50:110],
                         # Multiplication pour l'échelle
                         tauu_curl_change_array_MMM[50:110, 110:300] * 1e7,
                         transform=ccrs.PlateCarree(), cmap='RdBu_r', levels=levels2, extend="both")

cbar2 = fig.colorbar(contourf2, ax=ax1, orientation="horizontal",
                     fraction=0.046, pad=0.1, aspect=45)
cbar2.ax.tick_params(labelsize=55)
cbar2.set_label("Wind Stress Curl (10⁻⁷ N.m⁻³)", size=60, labelpad=15)

# Ajouter les boîtes de régions
ax1.add_geometries([geom_EAST], crs=ccrs.PlateCarree(),
                   facecolor="none", edgecolor="black", linewidth=4)
ax1.add_geometries([geom_WEST], crs=ccrs.PlateCarree(),
                   facecolor="none", edgecolor="black", linewidth=4)

# Ajouter les lignes de grille
gl2 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=1, linestyle='--')
gl2.xlocator = mticker.FixedLocator(
    [120, 140, 160, 180, -160, -140, -120, -100, -80])
gl2.ylocator = mticker.FixedLocator([-30, -20, -10, 0, 10, 20, 30])
gl2.top_labels = False
gl2.right_labels = False
gl2.xlabel_style = {'size': 45}
gl2.ylabel_style = {'size': 45}

ax1.coastlines(resolution='50m')
ax1.add_feature(cfeature.LAND, zorder=100, edgecolor="k")
ax1.set_title("(a)  MMM Wind Stress Curl",
              fontsize=65, fontweight='bold', pad=30)

# First line, second and third columns: Wind curl scatter plots
ylim = (-3, 3)
xlim_2 = (-0.175, 0.15)  # Ajustez selon l'échelle de votre curl

ax_scatter1 = create_scatter_plot(fig.add_subplot(gs[0, 1]),
                                  tauu_curl_CMIPS_WEST_change_mean * 1e7,  # Multiplication pour l'échelle
                                  Patterns_relative_D0_CMIPS_WEST_change_mean,
                                  "(b) Wind Curl vs ΔO'* in WEP",
                                  model_list_array, modele_couleur_marqueur,
                                  "Wind Stress Curl (10⁻⁷ N.m⁻³)",
                                  (0.05, 0.95), xlim_2, ylim)

ax_scatter1.set_ylabel("ΔO'*(°C)", fontsize=65)

ax_scatter2 = create_scatter_plot(fig.add_subplot(gs[0, 2]),
                                  tauu_curl_CMIPS_EAST_change_mean * 1e7,  # Multiplication pour l'échelle
                                  Patterns_relative_D0_CMIPS_EAST_change_mean,
                                  "(c) Wind Curl vs ΔO'* in EEP",
                                  model_list_array, modele_couleur_marqueur,
                                  "Wind Stress Curl (10⁻⁷ N.m⁻³)",
                                  (0.05, 0.95), xlim_2, ylim)

ax_scatter2.set_ylabel("ΔO'*(°C)", fontsize=65)

# Second line, first column: Temperature bias map
ax3 = fig.add_subplot(
    gs[1, 0], projection=ccrs.PlateCarree(central_longitude=180))

ccc1 = 3
step1 = ccc1 * 0.05
levels1 = np.arange(-ccc1, ccc1 + step1, step1)

contourf1 = ax3.contourf(lons_tos_2075_2100[110:300], lats_tos_2075_2100[50:110],
                         tos_biais_array_MMM[50:110, 110:300],
                         transform=ccrs.PlateCarree(), cmap='RdBu_r', levels=levels1, extend="both")

cbar1 = fig.colorbar(contourf1, ax=ax3, orientation="horizontal", label="°C",
                     fraction=0.046, pad=0.1, aspect=45)
cbar1.ax.tick_params(labelsize=55)
cbar1.set_label("°C", size=60, labelpad=15)

ax3.add_geometries([geom_EAST], crs=ccrs.PlateCarree(),
                   facecolor="none", edgecolor="black", linewidth=4)
ax3.add_geometries([geom_WEST], crs=ccrs.PlateCarree(),
                   facecolor="none", edgecolor="black", linewidth=4)

gl1 = ax3.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=1, linestyle='--')
gl1.xlocator = mticker.FixedLocator(
    [120, 140, 160, 180, -160, -140, -120, -100, -80])
gl1.ylocator = mticker.FixedLocator([-30, -20, -10, 0, 10, 20, 30])
gl1.top_labels = False
gl1.right_labels = False
gl1.xlabel_style = {'size': 45}
gl1.ylabel_style = {'size': 45}

ax3.coastlines(resolution='50m')
ax3.add_feature(cfeature.LAND, zorder=100, edgecolor="k")
ax3.set_title("(d)  MMM T bias", fontsize=65, fontweight='bold', pad=40)

# Second line, second and third columns: Temperature bias scatter plots
xlim_1 = (-4, 1)

ax_scatter3 = create_scatter_plot(fig.add_subplot(gs[1, 1]),
                                  tos_biais_CMIPS_WEST_change_mean,
                                  Patterns_relative_D0_CMIPS_WEST_change_mean,
                                  "(e) T bias vs ΔO'* in WEP",
                                  model_list_array, modele_couleur_marqueur,
                                  "T bias (°C)",
                                  (0.75, 0.1), xlim_1, ylim)

ax_scatter3.set_ylabel("ΔO'*(°C)", fontsize=65)

ax_scatter4 = create_scatter_plot(fig.add_subplot(gs[1, 2]),
                                  tos_biais_CMIPS_EAST_change_mean,
                                  Patterns_relative_D0_CMIPS_EAST_change_mean,
                                  "(f) T bias vs ΔO'* in EEP",
                                  model_list_array, modele_couleur_marqueur,
                                  "T bias (°C)",
                                  (0.75, 0.1), xlim_1, ylim)

ax_scatter4.set_ylabel("ΔO'*(°C)", fontsize=65)

# Adjust borders for all maps
for ax in [ax1, ax3]:
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
fig.savefig('your_path/figure_S4_curl.pdf',
            format='pdf', bbox_inches='tight')
fig.savefig('your_path/figure_S4_curl.png',
            format='png', dpi=80, bbox_inches='tight')

plt.show()


##################################################################### FIGURE S2


def calculate_relative_tos_changes_cmip5_cmip6():
    """
    Calcule les changements relatifs de température de surface 
    séparément pour CMIP5, CMIP6 et tous les modèles
    """
    global relative_tos_change_array_STD_cmip5
    global relative_tos_change_array_STD_cmip6
    global relative_tos_change_array_STD

    # Séparer les données pour CMIP5
    tos_change_array_cmip5 = tos_change_array[icmip5]

    # mean state calcul pour CMIP5
    tos_change_array_fitre_land_cmip5 = np.where(
        tos_change_array_cmip5 == 0, np.nan, tos_change_array_cmip5)
    tos_change_mean_state_lat_cmip5 = np.nanmean(
        tos_change_array_fitre_land_cmip5[:, 55:104, 0:360], axis=1)
    tos_change_mean_state_cmip5 = np.nanmean(
        tos_change_mean_state_lat_cmip5, axis=1)

    # Calcul des changements relatifs pour CMIP5
    relative_tos_change_list_cmip5 = []
    for i in range(len(icmip5)):
        relative_tos_change = tos_change_array_cmip5[i] - \
            tos_change_mean_state_cmip5[i]
        relative_tos_change_list_cmip5.append(relative_tos_change)

    relative_tos_change_array_cmip5 = np.array(relative_tos_change_list_cmip5)
    relative_tos_change_array_STD_cmip5 = np.nanstd(
        relative_tos_change_array_cmip5, axis=0)

    # Séparer les données pour CMIP6
    tos_change_array_cmip6 = tos_change_array[icmip6]

    # mean state calcul pour CMIP6
    tos_change_array_fitre_land_cmip6 = np.where(
        tos_change_array_cmip6 == 0, np.nan, tos_change_array_cmip6)
    tos_change_mean_state_lat_cmip6 = np.nanmean(
        tos_change_array_fitre_land_cmip6[:, 55:104, 0:360], axis=1)
    tos_change_mean_state_cmip6 = np.nanmean(
        tos_change_mean_state_lat_cmip6, axis=1)

    # Calcul des changements relatifs pour CMIP6
    relative_tos_change_list_cmip6 = []
    for i in range(len(icmip6)):
        relative_tos_change = tos_change_array_cmip6[i] - \
            tos_change_mean_state_cmip6[i]
        relative_tos_change_list_cmip6.append(relative_tos_change)

    relative_tos_change_array_cmip6 = np.array(relative_tos_change_list_cmip6)
    relative_tos_change_array_STD_cmip6 = np.nanstd(
        relative_tos_change_array_cmip6, axis=0)

    # Tous les modèles
    tos_change_array_fitre_land = np.where(
        tos_change_array == 0, np.nan, tos_change_array)
    tos_change_mean_state_lat = np.nanmean(
        tos_change_array_fitre_land[:, 55:104, 0:360], axis=1)
    tos_change_mean_state = np.nanmean(tos_change_mean_state_lat, axis=1)

    relative_tos_change_list = []
    for i in range(n):
        relative_tos_change = tos_change_array[i] - tos_change_mean_state[i]
        relative_tos_change_list.append(relative_tos_change)

    relative_tos_change_array = np.array(relative_tos_change_list)
    relative_tos_change_array_STD = np.nanstd(
        relative_tos_change_array, axis=0)

    return (relative_tos_change_array_cmip5, relative_tos_change_array_cmip6, relative_tos_change_array)


def create_rsst_std_figure():
    # Appeler la fonction pour calculer les changements relatifs
    relative_tos_change_array_cmip5, relative_tos_change_array_cmip6, relative_tos_change_array = calculate_relative_tos_changes_cmip5_cmip6()

    # Extraction des régions tropicales du Pacifique
    relative_tos_pacific_cmip5 = relative_tos_change_array_cmip5[:,
                                                                 50:110, 110:300]
    relative_tos_pacific_cmip6 = relative_tos_change_array_cmip6[:,
                                                                 50:110, 110:300]
    relative_tos_pacific = relative_tos_change_array[:, 50:110, 110:300]

    # Coordonnées pour le tracé
    lon_tropic = lons_tos_2075_2100[110:300]
    lat_tropic = lats_tos_2075_2100[50:110]

    # Créer la figure avec 3 lignes et 1 colonne
    fig, axs = plt.subplots(3, 1, figsize=(16, 18),
                            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

    # Définir les niveaux pour les cartes STD
    ccc_std = 0.6
    step_std = 0.025
    levels_std = np.arange(0, ccc_std + step_std, step_std)

    # ========================
    # CMIP5 STD
    # ========================
    contour1 = axs[0].contourf(lon_tropic, lat_tropic,
                               relative_tos_change_array_STD_cmip5[50:110, 110:300],
                               transform=ccrs.PlateCarree(), cmap='jet',
                               levels=levels_std, extend="max")

    # ========================
    # CMIP6 STD
    # ========================
    contour2 = axs[1].contourf(lon_tropic, lat_tropic,
                               relative_tos_change_array_STD_cmip6[50:110, 110:300],
                               transform=ccrs.PlateCarree(), cmap='jet',
                               levels=levels_std, extend="max")

    # ========================
    # All CMIP STD
    # ========================
    contour3 = axs[2].contourf(lon_tropic, lat_tropic,
                               relative_tos_change_array_STD[50:110, 110:300],
                               transform=ccrs.PlateCarree(), cmap='jet',
                               levels=levels_std, extend="max")

    # Ajouter les titres aux sous-graphiques
    axs[0].set_title("(a) CMIP5 STD of ΔT' change",
                     fontsize=20, fontweight='bold')
    axs[1].set_title("(b) CMIP6 STD of ΔT' change",
                     fontsize=20, fontweight='bold')
    axs[2].set_title("(c) All CMIP STD of ΔT' change",
                     fontsize=20, fontweight='bold')

    # Configuration de tous les sous-graphiques
    for i in range(3):
        # Ajouter les lignes de grille
        gl = axs[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator(
            [120, 140, 160, 180, -160, -140, -120, -100, -80])
        gl.ylocator = mticker.FixedLocator([-30, -20, -10, 0, 10, 20, 30])
        gl.top_labels = False
        gl.right_labels = False

        gl.xlabel_style = {'size': 12}
        gl.ylabel_style = {'size': 12}

        # Ajouter les bordures de continents et les terres
        axs[i].coastlines(resolution='50m')
        axs[i].add_feature(cfeature.LAND, zorder=100, edgecolor="k")

        # Ajouter les géométries de région
        axs[i].add_geometries([geom_SOUTH], crs=ccrs.PlateCarree(),
                              facecolor="none", edgecolor="black", linewidth=2)
        axs[i].add_geometries([geom_EAST], crs=ccrs.PlateCarree(),
                              facecolor="none", edgecolor="black", linewidth=2)
        axs[i].add_geometries([geom_WEST], crs=ccrs.PlateCarree(),
                              facecolor="none", edgecolor="black", linewidth=2)

        # Styliser les bordures
        for spine in axs[i].spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

    # Ajuster l'espacement pour laisser de la place à la barre de couleur horizontale
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, bottom=0.15)

    # Ajouter une barre de couleur commune horizontale en bas de la figure
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    cbar = fig.colorbar(contour1, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks(np.arange(0, 0.7, 0.1))
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label("°C", size=18)

    # Enregistrer la figure
    fig.savefig('your_path/figure_S2.pdf',
                format='pdf', bbox_inches='tight', dpi=100)
    fig.savefig('your_path/figure_S2.png',
                format='png', dpi=100, bbox_inches='tight')

    # Afficher la figure
    plt.show()

    return fig


# Exécuter le code
fig = create_rsst_std_figure()




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script simplifié pour exporter les températures futures dans l'ordre normalisé
"""


# "

# Chemins des fichiers
normalized_file = '/home/vincent/codes/programmes_thèses/EOF/zhang/before_Shakespeare/Open_acces/software/rankin_test/present_temperature_ranking.csv'
output_file = '/home/vincent/codes/programmes_thèses/EOF/zhang/before_Shakespeare/Open_acces/software/rankin_test/future_temperature_ranking.csv'

# Charger les données normalisées
normalized_data = pd.read_csv(normalized_file)
print(f"Données normalisées chargées : {len(normalized_data)} modèles")

# Récupérer les données futures (à partir des variables déjà chargées)
future_temps = RSST_somme_component_CMIPS_EAST_change_mean
model_list = model_list_array

# Créer le DataFrame des données futures
future_data = pd.DataFrame({
    'Model': model_list,
    'Temperature_change_East_Future': future_temps
})

# Fusionner les DataFrames en conservant l'ordre des données normalisées
# Cette étape garantit que les modèles sont dans le même ordre que le classement normalisé
result = pd.merge(
    normalized_data[['Model']],
    future_data,
    on='Model',
    how='left'
)

# Vérifier si des données sont manquantes
missing = result[result['Temperature_change_East_Future'].isna()]
if not missing.empty:
    print(f"ATTENTION: {len(missing)} modèles n'ont pas de données futures.")

# Sauvegarder le résultat
result.to_csv(output_file, index=False)
print(f"Fichier créé avec succès: {output_file}")
print("\nPremières lignes:")
print(result.head())

# " normalized

# Chemins des fichiers
normalized_file = '/home/vincent/codes/programmes_thèses/EOF/zhang/before_Shakespeare/Open_acces/software/rankin_test/normalized_present_temperature_ranking.csv'
output_file = '/home/vincent/codes/programmes_thèses/EOF/zhang/before_Shakespeare/Open_acces/software/rankin_test/normalized_future_temperature_ranking.csv'

# Charger les données normalisées
normalized_data = pd.read_csv(normalized_file)
print(f"Données normalisées chargées : {len(normalized_data)} modèles")

# Récupérer les données futures (à partir des variables déjà chargées)
future_temps = RSST_somme_component_CMIPS_EAST_change_mean
model_list = model_list_array

# Créer le DataFrame des données futures
future_data = pd.DataFrame({
    'Model': model_list,
    'Temperature_change_East_Future': future_temps
})

# Fusionner les DataFrames en conservant l'ordre des données normalisées
# Cette étape garantit que les modèles sont dans le même ordre que le classement normalisé
result = pd.merge(
    normalized_data[['Model']],
    future_data,
    on='Model',
    how='left'
)

# Vérifier si des données sont manquantes
missing = result[result['Temperature_change_East_Future'].isna()]
if not missing.empty:
    print(f"ATTENTION: {len(missing)} modèles n'ont pas de données futures.")

# Sauvegarder le résultat
result.to_csv(output_file, index=False)
print(f"Fichier créé avec succès: {output_file}")
print("\nPremières lignes:")
print(result.head())


###################################### FIGURE R2


# Définir la figure principale
fig, axs = plt.subplots(2, 1, figsize=(12, 10), subplot_kw={
                        'projection': ccrs.PlateCarree(central_longitude=180)})

# Calculer la carte de variance directe pour chaque point
direct_variance_map = np.nanvar(
    RSST_somme_component_change_ALPHA_MMM[:, 50:110, 110:300], axis=0)

# Initialiser la carte de somme des covariances
covariance_sum_map = np.zeros_like(direct_variance_map)

# Préparer les composantes pour la carte de covariance
all_regions_shapes = direct_variance_map.shape
n_models = RSST_somme_component_change_ALPHA_MMM.shape[0]

# Calcul de la somme des covariances pour chaque point
# On commence par créer un tableau 3D avec toutes les composantes
components_3d = np.array([
    RSST_somme_component_change_ALPHA_MMM[:, 50:110, 110:300],
    Patterns_relative_D0_change_array[:, 50:110, 110:300],
    oceanic_flux_deviation_array_fraction[:, 50:110, 110:300],
    Pattern_atm_somme_component_change[:, 50:110, 110:300]
])

# Calculer les moyennes multimodèles pour chaque composante
components_mmm = np.nanmean(components_3d, axis=1)

# Calculer les écarts par rapport aux moyennes
components_ecart = np.zeros_like(components_3d)
for i in range(components_3d.shape[0]):
    components_ecart[i] = components_3d[i] - components_mmm[i]

# Calcul de la somme des covariances pour chaque point
for i in range(all_regions_shapes[0]):  # latitudes
    for j in range(all_regions_shapes[1]):  # longitudes
        # Extraire les valeurs de ΔT' pour ce point
        delta_t_point = RSST_somme_component_change_ALPHA_MMM[:, 50+i, 110+j]

        # Calculer la somme des covariances pour ce point (en excluant la première composante qui est ΔT' lui-même)
        cov_sum = 0
        for k in range(1, components_3d.shape[0]):
            cov_sum += np.nansum(delta_t_point *
                                 components_ecart[k, :, i, j]) / n_models

        covariance_sum_map[i, j] = cov_sum

# Définir les niveaux pour le contour avec vérification de sécurité
max_val = max(np.nanmax(direct_variance_map), np.nanmax(covariance_sum_map))
levels = np.linspace(0, 0.4, 20)

# Tracer les cartes
for i, (ax, data, title) in enumerate(zip(axs,
                                          [direct_variance_map,
                                              covariance_sum_map],
                                          ["(a) Direct Variance of ΔT'", "(b) Sum of Covariances"])):
    # Configurer la carte
    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.LAND, zorder=100, edgecolor="k")

    # Tracer le contour
    cf = ax.contourf(lons_tos_2075_2100[110:300], lats_tos_2075_2100[50:110],
                     data, transform=ccrs.PlateCarree(),
                     cmap='viridis', levels=levels, extend="max")

    # Ajouter les boîtes des régions
    ax.add_geometries([geom_SOUTH], crs=ccrs.PlateCarree(),
                      facecolor="none", edgecolor="black", linewidth=2)
    ax.add_geometries([geom_EAST], crs=ccrs.PlateCarree(),
                      facecolor="none", edgecolor="black", linewidth=2)
    ax.add_geometries([geom_WEST], crs=ccrs.PlateCarree(),
                      facecolor="none", edgecolor="black", linewidth=2)

    # Configurer les lignes de grille
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator(
        [120, 140, 160, 180, -160, -140, -120, -100, -80])
    gl.ylocator = mticker.FixedLocator([-30, -20, -10, 0, 10, 20, 30])
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.top_labels = False
    gl.right_labels = False

    # Ajouter le titre
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Encadrer la carte
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)


# Ajouter la colorbar horizontale en bas avec 9 ticks (tous les 0.05 de 0 à 0.4)
cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])
cbar = fig.colorbar(cf, cax=cbar_ax, orientation="horizontal",
                    ticks=np.arange(0, 0.45, 0.05))
cbar.set_label("Variance (°C²)", fontsize=12)
cbar.ax.tick_params(labelsize=10)


# Ajuster la mise en page
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# Sauvegarder la figure
fig.savefig('your_path/variance_validation_simple.pdf',
            format='pdf', bbox_inches='tight', dpi=100)
fig.savefig('your_path/variance_validation_simple.png',
            format='png', dpi=100, bbox_inches='tight')

# Afficher la figure
plt.show()


#################################################################### figure S4 #################################


# Configuration des couleurs et noms
colors = ['#ff7f0e', '#2ca02c', '#d62728']
base_color = '#d62728'
n_shades = 3
color_shades = [plt.cm.Reds(i/n_shades) for i in range(1, n_shades+1)]
all_colors = colors + color_shades + ['#8B4513']


# Conversion des données (assurez-vous que ces fonctions et variables sont définies)
data_SOUTH_percentage = convert_to_percentages(data_SOUTH)
df_rsst_SOUTH_percentage = pd.DataFrame(data_SOUTH_percentage, index=[0])
df_melted_SOUTH_percentage = df_rsst_SOUTH_percentage.melt(
    var_name='Variable', value_name='Percentage')
# Liste des noms de colonnes pour les barplots

# Limites globales pour les axes y
global_y_min = -120
global_y_max = 200

# Fonction pour barplots de pourcentage


def create_subplot_percentage(ax, df_melted, title, show_ylabel=False):
    sns.barplot(x='Variable', y='Percentage',
                data=df_melted, ax=ax, palette=all_colors)

    ax.set_ylim(global_y_min, global_y_max)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax.vlines(x=[2.5], ymin=global_y_min, ymax=global_y_max,
              color='red', linestyle='solid', linewidth=5)

    for i, p in enumerate(ax.patches):
        height = p.get_height()
        if np.isnan(height):
            continue
        y_pos = height + 0.02 * \
            (global_y_max - global_y_min) if height > 0 else height - \
            0.02*(global_y_max - global_y_min)
        va = 'bottom' if height > 0 else 'top'
        ax.text(p.get_x() + p.get_width() / 2., y_pos, f'{height:.1f}%',
                ha='center', va=va, color='black', fontweight='bold', fontsize=30)

    ax.tick_params(axis='y', labelsize=35)

    # Toujours définir l'étiquette Y avec une taille de police plus grande
    if show_ylabel:
        ax.set_ylabel("%", fontsize=36, fontweight='bold')
    else:
        ax.set_ylabel("")

    ax.set_xticklabels(name_DF_list, fontsize=40,
                       fontweight='bold', rotation=40, ha='center')
    ax.set_title(title, fontsize=45, fontweight='bold', pad=25)
    ax.set_facecolor('white')
    ax.set_xlabel('')
    ax.grid(True, linestyle='--', linewidth=1, color='gray')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(3)


def create_combined_figure():
    # Configuration de la figure avec une disposition de deux lignes, trois colonnes
    fig = plt.figure(figsize=(45, 28), facecolor='white')
    # Utilisation d'un GridSpec avec 2 lignes, 3 colonnes
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])
    fig.subplots_adjust(top=0.95, wspace=0.3, hspace=0.3, bottom=0.1)

    # Préparer les données pour les barplots
    # Première ligne (SEP) - Ces fonctions doivent être définies dans votre code
    df_melted_SOUTH_cmip5_percentage, df_melted_SOUTH_cmip6_percentage = prepare_cmip_specific_data_south()

    # Première ligne, 3 barplots dans l'ordre CMIP5, CMIP6, All pour SEP
    # Barplot 1: Modèles CMIP5 uniquement (SEP)
    ax1 = plt.subplot(gs[0, 0])
    create_subplot_percentage(
        ax1, df_melted_SOUTH_cmip5_percentage, "(a) SEP - CMIP5 Models Only")

    # Barplot 2: Modèles CMIP6 uniquement (SEP)
    ax2 = plt.subplot(gs[0, 1])
    create_subplot_percentage(
        ax2, df_melted_SOUTH_cmip6_percentage, "(b) SEP - CMIP6 Models Only")

    # Barplot 3: Tous les modèles CMIP (SEP)
    ax3 = plt.subplot(gs[0, 2])
    create_subplot_percentage(
        ax3, df_melted_SOUTH_percentage, "(c) SEP - All CMIP Models")

    # Deuxième ligne, 3 barplots dans l'ordre CMIP5, CMIP6, All pour EEP-WEP (GRADIENT)
    # Barplot 4: Modèles CMIP5 uniquement (GRADIENT)
    ax4 = plt.subplot(gs[1, 0])
    create_subplot_percentage(
        ax4, df_melted_GRADIENT_CMIP5_percentage, "(d) EEP-WEP - CMIP5 only")

    # Barplot 5: Modèles CMIP6 uniquement (GRADIENT)
    ax5 = plt.subplot(gs[1, 1])
    create_subplot_percentage(
        ax5, df_melted_GRADIENT_CMIP6_percentage, "(e) EEP-WEP - CMIP6 only")

    # Barplot 6: Tous les modèles CMIP (GRADIENT)
    ax6 = plt.subplot(gs[1, 2])
    create_subplot_percentage(
        ax6, df_melted_GRADIENT_percentage, "(f) EEP-WEP - All CMIP models")

    # Ajuster le layout
    plt.tight_layout(pad=3.0)

    return fig


# Exécuter le code pour créer la figure combinée
fig = create_combined_figure()
plt.savefig('your_path/figure_combined_sep_gradient.pdf',
            format='pdf', bbox_inches='tight')
plt.show()
