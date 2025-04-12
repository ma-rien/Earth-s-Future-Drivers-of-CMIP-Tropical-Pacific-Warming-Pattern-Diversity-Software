#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 11:46:07 2025

@author: vincent
"""




# ###########################################################################"" CMIPS

# ################################################WEST
relative_tos_CMIPS_WEST_change_list=[]
for i in range (n):

    test=relative_tos_change_array[i,78:82,150:200]
    relative_tos_CMIPS_WEST_change_list.append(test)  
relative_tos_CMIPS_WEST_change_array=np.array(relative_tos_CMIPS_WEST_change_list)
    
relative_tos_CMIPS_WEST_change_zonal=np.nanmean(relative_tos_CMIPS_WEST_change_array,axis=1)   
relative_tos_CMIPS_WEST_change_mean=np.nanmean(relative_tos_CMIPS_WEST_change_zonal,axis=1)   

###########################
Patterns_atm_variable_total_CMIPS_WEST_change_list=[]

for j in range (len(Patterns_atm_variable_total_array)):
    atm_var=Patterns_atm_variable_total_array[j]
    relative_atm_variable_CMIPS_WEST_change_list=[] 
    
    for i in range (n):
        test=atm_var[i,78:82,150:200]
        relative_atm_variable_CMIPS_WEST_change_list.append(test)
    
    Patterns_atm_variable_total_CMIPS_WEST_change_list.append(relative_atm_variable_CMIPS_WEST_change_list)
Patterns_atm_variable_total_CMIPS_WEST_change_array=np.array(Patterns_atm_variable_total_CMIPS_WEST_change_list)  
Patterns_atm_variable_total_CMIPS_WEST_change_array_zonal=np.nanmean(Patterns_atm_variable_total_CMIPS_WEST_change_list,axis=2)
Patterns_atm_variable_total_CMIPS_WEST_change_array_mean=np.nanmean(Patterns_atm_variable_total_CMIPS_WEST_change_array_zonal,axis=2)


# ########################"

Pattern_sum_atm_component_fraction_CMIPS_WEST_change_list=[]
for i in range (n):

    test=Pattern_atm_somme_component_change[i,78:82,150:200]
    Pattern_sum_atm_component_fraction_CMIPS_WEST_change_list.append(test)  
Pattern_sum_atm_component_fraction_CMIPS_WEST_change_array=np.array(Pattern_sum_atm_component_fraction_CMIPS_WEST_change_list)
    
Pattern_sum_atm_component_fraction_CMIPS_WEST_change_zonal=np.nanmean(Pattern_sum_atm_component_fraction_CMIPS_WEST_change_array,axis=1)   
Pattern_sum_atm_component_fraction_CMIPS_WEST_change_mean=np.nanmean(Pattern_sum_atm_component_fraction_CMIPS_WEST_change_zonal,axis=1)   

# ##############
Patterns_relative_D0_CMIPS_WEST_change_list=[]
for i in range (n):

    test=Patterns_relative_D0_change_array[i,78:82,150:200]
    Patterns_relative_D0_CMIPS_WEST_change_list.append(test)  
Patterns_relative_D0_CMIPS_WEST_change_array=np.array(Patterns_relative_D0_CMIPS_WEST_change_list)
    
Patterns_relative_D0_CMIPS_WEST_change_zonal=np.nanmean(Patterns_relative_D0_CMIPS_WEST_change_array,axis=1)   
Patterns_relative_D0_CMIPS_WEST_change_mean=np.nanmean(Patterns_relative_D0_CMIPS_WEST_change_zonal,axis=1)   

###################



oceanic_flux_deviation_fraction_CMIPS_WEST_change_list=[]
for i in range (n):

    test=oceanic_flux_deviation_array_fraction[i,78:82,150:200]
    oceanic_flux_deviation_fraction_CMIPS_WEST_change_list.append(test)  
oceanic_flux_deviation_fraction_CMIPS_WEST_change_array=np.array(oceanic_flux_deviation_fraction_CMIPS_WEST_change_list)
    
oceanic_flux_deviation_fraction_CMIPS_WEST_change_zonal=np.nanmean(oceanic_flux_deviation_fraction_CMIPS_WEST_change_array,axis=1)   
oceanic_flux_deviation_fraction_CMIPS_WEST_change_mean=np.nanmean(oceanic_flux_deviation_fraction_CMIPS_WEST_change_zonal,axis=1)




Spatial_heterogeneity_fraction_array_CMIPS_WEST_change_list=[]
for i in range (n):

    test=Spatial_heterogeneity_fraction_array[i,78:82,150:200]
    Spatial_heterogeneity_fraction_array_CMIPS_WEST_change_list.append(test)  
Spatial_heterogeneity_fraction_array_CMIPS_WEST_change_array=np.array(Spatial_heterogeneity_fraction_array_CMIPS_WEST_change_list)
    
Spatial_heterogeneity_fraction_array_CMIPS_WEST_change_zonal=np.nanmean(Spatial_heterogeneity_fraction_array_CMIPS_WEST_change_array,axis=1)   
Spatial_heterogeneity_fraction_array_CMIPS_WEST_change_mean=np.nanmean(Spatial_heterogeneity_fraction_array_CMIPS_WEST_change_zonal,axis=1)   





RSST_somme_component_CMIPS_WEST_change_list=[]
for i in range (n):

    test=RSST_somme_component_change_ALPHA_MMM[i,78:82,150:200]
    RSST_somme_component_CMIPS_WEST_change_list.append(test)
    RSST_somme_component_CMIPS_WEST_change=np.array(RSST_somme_component_CMIPS_WEST_change_list)
RSST_somme_component_CMIPS_WEST_change_zonal=np.nanmean(RSST_somme_component_CMIPS_WEST_change,axis=1)   
RSST_somme_component_CMIPS_WEST_change_mean=np.nanmean(RSST_somme_component_CMIPS_WEST_change_zonal,axis=1)   



###########################"



relative_sum_oceanic_component_fraction_CMIPS_WEST_change_list=[]
for i in range (n):

    test=oceanic_somme_component_change[i,78:82,150:200]
    relative_sum_oceanic_component_fraction_CMIPS_WEST_change_list.append(test)  
relative_sum_oceanic_component_fraction_CMIPS_WEST_change_array=np.array(relative_sum_oceanic_component_fraction_CMIPS_WEST_change_list)
    
relative_sum_oceanic_component_fraction_CMIPS_WEST_change_zonal=np.nanmean(relative_sum_oceanic_component_fraction_CMIPS_WEST_change_array,axis=1)   
relative_sum_oceanic_component_fraction_CMIPS_WEST_change_mean=np.nanmean(relative_sum_oceanic_component_fraction_CMIPS_WEST_change_zonal,axis=1)   















# ########### EAST 

# ################################################EAST
relative_tos_CMIPS_EAST_change_list=[]
for i in range (n):

    test=relative_tos_change_array[i,78:82,220:270]
    relative_tos_CMIPS_EAST_change_list.append(test)  
relative_tos_CMIPS_EAST_change_array=np.array(relative_tos_CMIPS_EAST_change_list)
    
relative_tos_CMIPS_EAST_change_zonal=np.nanmean(relative_tos_CMIPS_EAST_change_array,axis=1)   
relative_tos_CMIPS_EAST_change_mean=np.nanmean(relative_tos_CMIPS_EAST_change_zonal,axis=1)   

###########################
Patterns_atm_variable_total_CMIPS_EAST_change_list=[]

for j in range (len(Patterns_atm_variable_total_array)):
    atm_var=Patterns_atm_variable_total_array[j]
    relative_atm_variable_CMIPS_EAST_change_list=[] 
    
    for i in range (n):
        test=atm_var[i,78:82,220:270]
        relative_atm_variable_CMIPS_EAST_change_list.append(test)
    
    Patterns_atm_variable_total_CMIPS_EAST_change_list.append(relative_atm_variable_CMIPS_EAST_change_list)
Patterns_atm_variable_total_CMIPS_EAST_change_array=np.array(Patterns_atm_variable_total_CMIPS_EAST_change_list)  
Patterns_atm_variable_total_CMIPS_EAST_change_array_zonal=np.nanmean(Patterns_atm_variable_total_CMIPS_EAST_change_list,axis=2)
Patterns_atm_variable_total_CMIPS_EAST_change_array_mean=np.nanmean(Patterns_atm_variable_total_CMIPS_EAST_change_array_zonal,axis=2)


########################"

Pattern_sum_atm_component_fraction_CMIPS_EAST_change_list=[]
for i in range (n):

    test=Pattern_atm_somme_component_change[i,78:82,220:270]
    Pattern_sum_atm_component_fraction_CMIPS_EAST_change_list.append(test)  
Pattern_sum_atm_component_fraction_CMIPS_EAST_change_array=np.array(Pattern_sum_atm_component_fraction_CMIPS_EAST_change_list)
    
Pattern_sum_atm_component_fraction_CMIPS_EAST_change_zonal=np.nanmean(Pattern_sum_atm_component_fraction_CMIPS_EAST_change_array,axis=1)   
Pattern_sum_atm_component_fraction_CMIPS_EAST_change_mean=np.nanmean(Pattern_sum_atm_component_fraction_CMIPS_EAST_change_zonal,axis=1)   

##############
Patterns_relative_D0_CMIPS_EAST_change_list=[]
for i in range (n):

    test=Patterns_relative_D0_change_array[i,78:82,220:270]
    Patterns_relative_D0_CMIPS_EAST_change_list.append(test)  
Patterns_relative_D0_CMIPS_EAST_change_array=np.array(Patterns_relative_D0_CMIPS_EAST_change_list)
    
Patterns_relative_D0_CMIPS_EAST_change_zonal=np.nanmean(Patterns_relative_D0_CMIPS_EAST_change_array,axis=1)   
Patterns_relative_D0_CMIPS_EAST_change_mean=np.nanmean(Patterns_relative_D0_CMIPS_EAST_change_zonal,axis=1)   

###################



oceanic_flux_deviation_fraction_CMIPS_EAST_change_list=[]
for i in range (n):

    test=oceanic_flux_deviation_array_fraction[i,78:82,220:270]
    oceanic_flux_deviation_fraction_CMIPS_EAST_change_list.append(test)  
oceanic_flux_deviation_fraction_CMIPS_EAST_change_array=np.array(oceanic_flux_deviation_fraction_CMIPS_EAST_change_list)
    
oceanic_flux_deviation_fraction_CMIPS_EAST_change_zonal=np.nanmean(oceanic_flux_deviation_fraction_CMIPS_EAST_change_array,axis=1)   
oceanic_flux_deviation_fraction_CMIPS_EAST_change_mean=np.nanmean(oceanic_flux_deviation_fraction_CMIPS_EAST_change_zonal,axis=1)



Spatial_heterogeneity_fraction_array_CMIPS_EAST_change_list=[]
for i in range (n):

    test=Spatial_heterogeneity_fraction_array[i,78:82,220:270]
    Spatial_heterogeneity_fraction_array_CMIPS_EAST_change_list.append(test)  
Spatial_heterogeneity_fraction_array_CMIPS_EAST_change_array=np.array(Spatial_heterogeneity_fraction_array_CMIPS_EAST_change_list)
    
Spatial_heterogeneity_fraction_array_CMIPS_EAST_change_zonal=np.nanmean(Spatial_heterogeneity_fraction_array_CMIPS_EAST_change_array,axis=1)   
Spatial_heterogeneity_fraction_array_CMIPS_EAST_change_mean=np.nanmean(Spatial_heterogeneity_fraction_array_CMIPS_EAST_change_zonal,axis=1)   



###########################"



relative_sum_oceanic_component_fraction_CMIPS_EAST_change_list=[]
for i in range (n):

    test=(oceanic_somme_component_change[i,78:82,220:270])
    relative_sum_oceanic_component_fraction_CMIPS_EAST_change_list.append(test)  
relative_sum_oceanic_component_fraction_CMIPS_EAST_change_array=np.array(relative_sum_oceanic_component_fraction_CMIPS_EAST_change_list)
    
relative_sum_oceanic_component_fraction_CMIPS_EAST_change_zonal=np.nanmean(relative_sum_oceanic_component_fraction_CMIPS_EAST_change_array,axis=1)   
relative_sum_oceanic_component_fraction_CMIPS_EAST_change_mean=np.nanmean(relative_sum_oceanic_component_fraction_CMIPS_EAST_change_zonal,axis=1)   





RSST_somme_component_CMIPS_EAST_change_list=[]
for i in range (n):

    test=RSST_somme_component_change_ALPHA_MMM[i,78:82,220:270]
    RSST_somme_component_CMIPS_EAST_change_list.append(test)
    RSST_somme_component_CMIPS_EAST_change=np.array(RSST_somme_component_CMIPS_EAST_change_list)
RSST_somme_component_CMIPS_EAST_change_zonal=np.nanmean(RSST_somme_component_CMIPS_EAST_change,axis=1)   
RSST_somme_component_CMIPS_EAST_change_mean=np.nanmean(RSST_somme_component_CMIPS_EAST_change_zonal,axis=1)   







# ################################################NORTH
# relative_tos_CMIPS_NORTH_change_list=[]
# for i in range (n):

#     test=relative_tos_change_array[i,100:110,180:230]
#     relative_tos_CMIPS_NORTH_change_list.append(test)  
# relative_tos_CMIPS_NORTH_change_array=np.array(relative_tos_CMIPS_NORTH_change_list)
    
# relative_tos_CMIPS_NORTH_change_zonal=np.nanmean(relative_tos_CMIPS_NORTH_change_array,axis=1)   
# relative_tos_CMIPS_NORTH_change_mean=np.nanmean(relative_tos_CMIPS_NORTH_change_zonal,axis=1)   

###########################
Patterns_atm_variable_total_CMIPS_NORTH_change_list=[]

for j in range (len(Patterns_atm_variable_total_array)):
    atm_var=Patterns_atm_variable_total_array[j]
    relative_atm_variable_CMIPS_NORTH_change_list=[] 
    
    for i in range (n):
        test=atm_var[i,100:110,180:230]
        relative_atm_variable_CMIPS_NORTH_change_list.append(test)
    
    Patterns_atm_variable_total_CMIPS_NORTH_change_list.append(relative_atm_variable_CMIPS_NORTH_change_list)
Patterns_atm_variable_total_CMIPS_NORTH_change_array=np.array(Patterns_atm_variable_total_CMIPS_NORTH_change_list)  
Patterns_atm_variable_total_CMIPS_NORTH_change_array_zonal=np.nanmean(Patterns_atm_variable_total_CMIPS_NORTH_change_list,axis=2)
Patterns_atm_variable_total_CMIPS_NORTH_change_array_mean=np.nanmean(Patterns_atm_variable_total_CMIPS_NORTH_change_array_zonal,axis=2)


########################"

# Pattern_sum_atm_component_fraction_CMIPS_NORTH_change_list=[]
# for i in range (n):

#     test=Pattern_atm_somme_component_change[i,100:110,180:230]
#     Pattern_sum_atm_component_fraction_CMIPS_NORTH_change_list.append(test)  
# Pattern_sum_atm_component_fraction_CMIPS_NORTH_change_array=np.array(Pattern_sum_atm_component_fraction_CMIPS_NORTH_change_list)
    
# Pattern_sum_atm_component_fraction_CMIPS_NORTH_change_zonal=np.nanmean(Pattern_sum_atm_component_fraction_CMIPS_NORTH_change_array,axis=1)   
# Pattern_sum_atm_component_fraction_CMIPS_NORTH_change_mean=np.nanmean(Pattern_sum_atm_component_fraction_CMIPS_NORTH_change_zonal,axis=1)   

##############
Patterns_relative_D0_CMIPS_NORTH_change_list=[]
for i in range (n):

    test=Patterns_relative_D0_change_array[i,100:110,180:230]
    Patterns_relative_D0_CMIPS_NORTH_change_list.append(test)  
Patterns_relative_D0_CMIPS_NORTH_change_array=np.array(Patterns_relative_D0_CMIPS_NORTH_change_list)
    
Patterns_relative_D0_CMIPS_NORTH_change_zonal=np.nanmean(Patterns_relative_D0_CMIPS_NORTH_change_array,axis=1)   
Patterns_relative_D0_CMIPS_NORTH_change_mean=np.nanmean(Patterns_relative_D0_CMIPS_NORTH_change_zonal,axis=1)   

###################



oceanic_flux_deviation_fraction_CMIPS_NORTH_change_list=[]
for i in range (n):

    test=oceanic_flux_deviation_array_fraction[i,100:110,180:230]
    oceanic_flux_deviation_fraction_CMIPS_NORTH_change_list.append(test)  
oceanic_flux_deviation_fraction_CMIPS_NORTH_change_array=np.array(oceanic_flux_deviation_fraction_CMIPS_NORTH_change_list)
    
oceanic_flux_deviation_fraction_CMIPS_NORTH_change_zonal=np.nanmean(oceanic_flux_deviation_fraction_CMIPS_NORTH_change_array,axis=1)   
oceanic_flux_deviation_fraction_CMIPS_NORTH_change_mean=np.nanmean(oceanic_flux_deviation_fraction_CMIPS_NORTH_change_zonal,axis=1)




Spatial_heterogeneity_fraction_array_CMIPS_NORTH_change_list=[]
for i in range (n):

    test=Spatial_heterogeneity_fraction_array[i,100:110,180:230]
    Spatial_heterogeneity_fraction_array_CMIPS_NORTH_change_list.append(test)  
Spatial_heterogeneity_fraction_array_CMIPS_NORTH_change_array=np.array(Spatial_heterogeneity_fraction_array_CMIPS_NORTH_change_list)
    
Spatial_heterogeneity_fraction_array_CMIPS_NORTH_change_zonal=np.nanmean(Spatial_heterogeneity_fraction_array_CMIPS_NORTH_change_array,axis=1)   
Spatial_heterogeneity_fraction_array_CMIPS_NORTH_change_mean=np.nanmean(Spatial_heterogeneity_fraction_array_CMIPS_NORTH_change_zonal,axis=1)   



###########################"



relative_sum_oceanic_component_fraction_CMIPS_NORTH_change_list=[]
for i in range (n):

    test=oceanic_somme_component_change[i,100:110,180:230]
    relative_sum_oceanic_component_fraction_CMIPS_NORTH_change_list.append(test)  
relative_sum_oceanic_component_fraction_CMIPS_NORTH_change_array=np.array(relative_sum_oceanic_component_fraction_CMIPS_NORTH_change_list)
    
relative_sum_oceanic_component_fraction_CMIPS_NORTH_change_zonal=np.nanmean(relative_sum_oceanic_component_fraction_CMIPS_NORTH_change_array,axis=1)   
relative_sum_oceanic_component_fraction_CMIPS_NORTH_change_mean=np.nanmean(relative_sum_oceanic_component_fraction_CMIPS_NORTH_change_zonal,axis=1)   





RSST_somme_component_CMIPS_NORTH_change_list=[]
for i in range (n):

    test=RSST_somme_component_change_ALPHA_MMM[i,100:110,180:230]
    RSST_somme_component_CMIPS_NORTH_change_list.append(test)
    RSST_somme_component_CMIPS_NORTH_change=np.array(RSST_somme_component_CMIPS_NORTH_change_list)
RSST_somme_component_CMIPS_NORTH_change_zonal=np.nanmean(RSST_somme_component_CMIPS_NORTH_change,axis=1)   
RSST_somme_component_CMIPS_NORTH_change_mean=np.nanmean(RSST_somme_component_CMIPS_NORTH_change_zonal,axis=1)   







#################################################################SOUTH
relative_tos_CMIPS_SOUTH_change_list=[]
for i in range (n):

    test=relative_tos_change_array[i,57:67,230:280]
    relative_tos_CMIPS_SOUTH_change_list.append(test)  
relative_tos_CMIPS_SOUTH_change_array=np.array(relative_tos_CMIPS_SOUTH_change_list)
    
relative_tos_CMIPS_SOUTH_change_zonal=np.nanmean(relative_tos_CMIPS_SOUTH_change_array,axis=1)   
relative_tos_CMIPS_SOUTH_change_mean=np.nanmean(relative_tos_CMIPS_SOUTH_change_zonal,axis=1)   

###########################
Patterns_atm_variable_total_CMIPS_SOUTH_change_list=[]

for j in range (len(Patterns_atm_variable_total_array)):
    atm_var=Patterns_atm_variable_total_array[j]
    relative_atm_variable_CMIPS_SOUTH_change_list=[] 
    
    for i in range (n):
        test=atm_var[i,57:67,230:280]
        relative_atm_variable_CMIPS_SOUTH_change_list.append(test)
    
    Patterns_atm_variable_total_CMIPS_SOUTH_change_list.append(relative_atm_variable_CMIPS_SOUTH_change_list)
Patterns_atm_variable_total_CMIPS_SOUTH_change_array=np.array(Patterns_atm_variable_total_CMIPS_SOUTH_change_list)  
Patterns_atm_variable_total_CMIPS_SOUTH_change_array_zonal=np.nanmean(Patterns_atm_variable_total_CMIPS_SOUTH_change_list,axis=2)
Patterns_atm_variable_total_CMIPS_SOUTH_change_array_mean=np.nanmean(Patterns_atm_variable_total_CMIPS_SOUTH_change_array_zonal,axis=2)


########################"

Pattern_sum_atm_component_fraction_CMIPS_SOUTH_change_list=[]
for i in range (n):

    test=Pattern_atm_somme_component_change[i,57:67,230:280]
    Pattern_sum_atm_component_fraction_CMIPS_SOUTH_change_list.append(test)  
Pattern_sum_atm_component_fraction_CMIPS_SOUTH_change_array=np.array(Pattern_sum_atm_component_fraction_CMIPS_SOUTH_change_list)
    
Pattern_sum_atm_component_fraction_CMIPS_SOUTH_change_zonal=np.nanmean(Pattern_sum_atm_component_fraction_CMIPS_SOUTH_change_array,axis=1)   
Pattern_sum_atm_component_fraction_CMIPS_SOUTH_change_mean=np.nanmean(Pattern_sum_atm_component_fraction_CMIPS_SOUTH_change_zonal,axis=1)   

# ##############
Patterns_relative_D0_CMIPS_SOUTH_change_list=[]
for i in range (n):

    test=Patterns_relative_D0_change_array[i,57:67,230:280]
    Patterns_relative_D0_CMIPS_SOUTH_change_list.append(test)  
Patterns_relative_D0_CMIPS_SOUTH_change_array=np.array(Patterns_relative_D0_CMIPS_SOUTH_change_list)
    
Patterns_relative_D0_CMIPS_SOUTH_change_zonal=np.nanmean(Patterns_relative_D0_CMIPS_SOUTH_change_array,axis=1)   
Patterns_relative_D0_CMIPS_SOUTH_change_mean=np.nanmean(Patterns_relative_D0_CMIPS_SOUTH_change_zonal,axis=1)   

###################



oceanic_flux_deviation_fraction_CMIPS_SOUTH_change_list=[]
for i in range (n):

    test=oceanic_flux_deviation_array_fraction[i,57:67,230:280]
    oceanic_flux_deviation_fraction_CMIPS_SOUTH_change_list.append(test)  
oceanic_flux_deviation_fraction_CMIPS_SOUTH_change_array=np.array(oceanic_flux_deviation_fraction_CMIPS_SOUTH_change_list)
    
oceanic_flux_deviation_fraction_CMIPS_SOUTH_change_zonal=np.nanmean(oceanic_flux_deviation_fraction_CMIPS_SOUTH_change_array,axis=1)   
oceanic_flux_deviation_fraction_CMIPS_SOUTH_change_mean=np.nanmean(oceanic_flux_deviation_fraction_CMIPS_SOUTH_change_zonal,axis=1)



Spatial_heterogeneity_fraction_array_CMIPS_SOUTH_change_list=[]
for i in range (n):

    test=Spatial_heterogeneity_fraction_array[i,57:67,230:280]
    Spatial_heterogeneity_fraction_array_CMIPS_SOUTH_change_list.append(test)  
Spatial_heterogeneity_fraction_array_CMIPS_SOUTH_change_array=np.array(Spatial_heterogeneity_fraction_array_CMIPS_SOUTH_change_list)
    
Spatial_heterogeneity_fraction_array_CMIPS_SOUTH_change_zonal=np.nanmean(Spatial_heterogeneity_fraction_array_CMIPS_SOUTH_change_array,axis=1)   
Spatial_heterogeneity_fraction_array_CMIPS_SOUTH_change_mean=np.nanmean(Spatial_heterogeneity_fraction_array_CMIPS_SOUTH_change_zonal,axis=1)   

###########################"



relative_sum_oceanic_component_fraction_CMIPS_SOUTH_change_list=[]
for i in range (n):

    test=oceanic_somme_component_change[i,57:67,230:280]
    relative_sum_oceanic_component_fraction_CMIPS_SOUTH_change_list.append(test)  
relative_sum_oceanic_component_fraction_CMIPS_SOUTH_change_array=np.array(relative_sum_oceanic_component_fraction_CMIPS_SOUTH_change_list)
    
relative_sum_oceanic_component_fraction_CMIPS_SOUTH_change_zonal=np.nanmean(relative_sum_oceanic_component_fraction_CMIPS_SOUTH_change_array,axis=1)   
relative_sum_oceanic_component_fraction_CMIPS_SOUTH_change_mean=np.nanmean(relative_sum_oceanic_component_fraction_CMIPS_SOUTH_change_zonal,axis=1)   




RSST_somme_component_CMIPS_SOUTH_change_list=[]
for i in range (n):

    test=RSST_somme_component_change_ALPHA_MMM[i,57:67,230:280]
    RSST_somme_component_CMIPS_SOUTH_change_list.append(test)
    RSST_somme_component_CMIPS_SOUTH_change=np.array(RSST_somme_component_CMIPS_SOUTH_change_list)
RSST_somme_component_CMIPS_SOUTH_change_zonal=np.nanmean(RSST_somme_component_CMIPS_SOUTH_change,axis=1)   
RSST_somme_component_CMIPS_SOUTH_change_mean=np.nanmean(RSST_somme_component_CMIPS_SOUTH_change_zonal,axis=1)   




##################"


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 19:22:26 2023

@author: vincent
"""












# ################################################EQUATOR
relative_tos_CMIPS_EQUATOR_change_list=[]
for i in range (n):

    test=relative_tos_change_array[i,78:82,150:270]
    relative_tos_CMIPS_EQUATOR_change_list.append(test)  
relative_tos_CMIPS_EQUATOR_change_array=np.array(relative_tos_CMIPS_EQUATOR_change_list)
    
relative_tos_CMIPS_EQUATOR_change_zonal=np.nanmean(relative_tos_CMIPS_EQUATOR_change_array,axis=1)   
relative_tos_CMIPS_EQUATOR_change_mean=np.nanmean(relative_tos_CMIPS_EQUATOR_change_zonal,axis=1)   

###########################
Patterns_atm_variable_total_CMIPS_EQUATOR_change_list=[]

for j in range (len(Patterns_atm_variable_total_array)):
    atm_var=Patterns_atm_variable_total_array[j]
    relative_atm_variable_CMIPS_EQUATOR_change_list=[] 
    
    for i in range (n):
        test=atm_var[i,78:82,150:270]
        relative_atm_variable_CMIPS_EQUATOR_change_list.append(test)
    
    Patterns_atm_variable_total_CMIPS_EQUATOR_change_list.append(relative_atm_variable_CMIPS_EQUATOR_change_list)
Patterns_atm_variable_total_CMIPS_EQUATOR_change_array=np.array(Patterns_atm_variable_total_CMIPS_EQUATOR_change_list)  
Patterns_atm_variable_total_CMIPS_EQUATOR_change_array_zonal=np.nanmean(Patterns_atm_variable_total_CMIPS_EQUATOR_change_list,axis=2)
Patterns_atm_variable_total_CMIPS_EQUATOR_change_array_mean=np.nanmean(Patterns_atm_variable_total_CMIPS_EQUATOR_change_array_zonal,axis=2)


########################"

# Pattern_sum_atm_component_fraction_CMIPS_EQUATOR_change_list=[]
# for i in range (n):

#     test=Pattern_atm_somme_component_change[i,78:82,150:270]
#     Pattern_sum_atm_component_fraction_CMIPS_EQUATOR_change_list.append(test)  
# Pattern_sum_atm_component_fraction_CMIPS_EQUATOR_change_array=np.array(Pattern_sum_atm_component_fraction_CMIPS_EQUATOR_change_list)
    
# Pattern_sum_atm_component_fraction_CMIPS_EQUATOR_change_zonal=np.nanmean(Pattern_sum_atm_component_fraction_CMIPS_EQUATOR_change_array,axis=1)   
# Pattern_sum_atm_component_fraction_CMIPS_EQUATOR_change_mean=np.nanmean(Pattern_sum_atm_component_fraction_CMIPS_EQUATOR_change_zonal,axis=1)   

##############
Patterns_relative_D0_CMIPS_EQUATOR_change_list=[]
for i in range (n):

    test=Patterns_relative_D0_change_array[i,78:82,150:270]
    Patterns_relative_D0_CMIPS_EQUATOR_change_list.append(test)  
Patterns_relative_D0_CMIPS_EQUATOR_change_array=np.array(Patterns_relative_D0_CMIPS_EQUATOR_change_list)
    
Patterns_relative_D0_CMIPS_EQUATOR_change_zonal=np.nanmean(Patterns_relative_D0_CMIPS_EQUATOR_change_array,axis=1)   
Patterns_relative_D0_CMIPS_EQUATOR_change_mean=np.nanmean(Patterns_relative_D0_CMIPS_EQUATOR_change_zonal,axis=1)   

###################



oceanic_flux_deviation_fraction_CMIPS_EQUATOR_change_list=[]
for i in range (n):

    test=oceanic_flux_deviation_array_fraction[i,78:82,150:270]
    oceanic_flux_deviation_fraction_CMIPS_EQUATOR_change_list.append(test)  
oceanic_flux_deviation_fraction_CMIPS_EQUATOR_change_array=np.array(oceanic_flux_deviation_fraction_CMIPS_EQUATOR_change_list)
    
oceanic_flux_deviation_fraction_CMIPS_EQUATOR_change_zonal=np.nanmean(oceanic_flux_deviation_fraction_CMIPS_EQUATOR_change_array,axis=1)   
oceanic_flux_deviation_fraction_CMIPS_EQUATOR_change_mean=np.nanmean(oceanic_flux_deviation_fraction_CMIPS_EQUATOR_change_zonal,axis=1)




Spatial_heterogeneity_fraction_array_CMIPS_EQUATOR_change_list=[]
for i in range (n):

    test=Spatial_heterogeneity_fraction_array[i,78:82,150:270]
    Spatial_heterogeneity_fraction_array_CMIPS_EQUATOR_change_list.append(test)  
Spatial_heterogeneity_fraction_array_CMIPS_EQUATOR_change_array=np.array(Spatial_heterogeneity_fraction_array_CMIPS_EQUATOR_change_list)
    
Spatial_heterogeneity_fraction_array_CMIPS_EQUATOR_change_zonal=np.nanmean(Spatial_heterogeneity_fraction_array_CMIPS_EQUATOR_change_array,axis=1)   
Spatial_heterogeneity_fraction_array_CMIPS_EQUATOR_change_mean=np.nanmean(Spatial_heterogeneity_fraction_array_CMIPS_EQUATOR_change_zonal,axis=1)   






###########################"



relative_sum_oceanic_component_fraction_CMIPS_EQUATOR_change_list=[]
for i in range (n):

    test=oceanic_somme_component_change[i,78:82,150:270]
    relative_sum_oceanic_component_fraction_CMIPS_EQUATOR_change_list.append(test)  
relative_sum_oceanic_component_fraction_CMIPS_EQUATOR_change_array=np.array(relative_sum_oceanic_component_fraction_CMIPS_EQUATOR_change_list)
    
relative_sum_oceanic_component_fraction_CMIPS_EQUATOR_change_zonal=np.nanmean(relative_sum_oceanic_component_fraction_CMIPS_EQUATOR_change_array,axis=1)   
relative_sum_oceanic_component_fraction_CMIPS_EQUATOR_change_mean=np.nanmean(relative_sum_oceanic_component_fraction_CMIPS_EQUATOR_change_zonal,axis=1)   






RSST_somme_component_CMIPS_EQUATOR_change_list=[]
for i in range (n):

    test=RSST_somme_component_change_ALPHA_MMM[i,78:82,150:270]
    RSST_somme_component_CMIPS_EQUATOR_change_list.append(test)
    RSST_somme_component_CMIPS_EQUATOR_change=np.array(RSST_somme_component_CMIPS_EQUATOR_change_list)
RSST_somme_component_CMIPS_EQUATOR_change_zonal=np.nanmean(RSST_somme_component_CMIPS_EQUATOR_change,axis=1)   
RSST_somme_component_CMIPS_EQUATOR_change_mean=np.nanmean(RSST_somme_component_CMIPS_EQUATOR_change_zonal,axis=1)   




############################### relative




relative_atm_variable_total_CMIPS_SOUTH_change_list=[]

for j in range (len(relative_atm_variable_array)):
    atm_var=relative_atm_variable_array[j]
    relative_atm_variable_CMIPS_SOUTH_change_list=[] 
    
    for i in range (n):
        test=atm_var[i,57:67,230:280]
        relative_atm_variable_CMIPS_SOUTH_change_list.append(test)
    
    relative_atm_variable_total_CMIPS_SOUTH_change_list.append(relative_atm_variable_CMIPS_SOUTH_change_list)
relative_atm_variable_total_CMIPS_SOUTH_change_array=np.array(relative_atm_variable_total_CMIPS_SOUTH_change_list)  
relative_atm_variable_total_CMIPS_SOUTH_change_array_zonal=np.nanmean(relative_atm_variable_total_CMIPS_SOUTH_change_list,axis=2)
relative_atm_variable_total_CMIPS_SOUTH_change_array_mean=np.nanmean(relative_atm_variable_total_CMIPS_SOUTH_change_array_zonal,axis=2)







# ################################################COLD
relative_tos_CMIPS_COLD_change_list=[]
for i in range (n):

    test=relative_tos_change_array[i,78:82,180:250]
    relative_tos_CMIPS_COLD_change_list.append(test)  
relative_tos_CMIPS_COLD_change_array=np.array(relative_tos_CMIPS_COLD_change_list)
    
relative_tos_CMIPS_COLD_change_zonal=np.nanmean(relative_tos_CMIPS_COLD_change_array,axis=1)   
relative_tos_CMIPS_COLD_change_mean=np.nanmean(relative_tos_CMIPS_COLD_change_zonal,axis=1)   

###########################
Patterns_atm_variable_total_CMIPS_COLD_change_list=[]

for j in range (len(Patterns_atm_variable_total_array)):
    atm_var=Patterns_atm_variable_total_array[j]
    relative_atm_variable_CMIPS_COLD_change_list=[] 
    
    for i in range (n):
        test=atm_var[i,78:82,180:250]
        relative_atm_variable_CMIPS_COLD_change_list.append(test)
    
    Patterns_atm_variable_total_CMIPS_COLD_change_list.append(relative_atm_variable_CMIPS_COLD_change_list)
Patterns_atm_variable_total_CMIPS_COLD_change_array=np.array(Patterns_atm_variable_total_CMIPS_COLD_change_list)  
Patterns_atm_variable_total_CMIPS_COLD_change_array_zonal=np.nanmean(Patterns_atm_variable_total_CMIPS_COLD_change_list,axis=2)
Patterns_atm_variable_total_CMIPS_COLD_change_array_mean=np.nanmean(Patterns_atm_variable_total_CMIPS_COLD_change_array_zonal,axis=2)


########################"

Pattern_sum_atm_component_fraction_CMIPS_COLD_change_list=[]
for i in range (n):

    test=Pattern_atm_somme_component_change[i,78:82,180:250]
    Pattern_sum_atm_component_fraction_CMIPS_COLD_change_list.append(test)  
Pattern_sum_atm_component_fraction_CMIPS_COLD_change_array=np.array(Pattern_sum_atm_component_fraction_CMIPS_COLD_change_list)
    
Pattern_sum_atm_component_fraction_CMIPS_COLD_change_zonal=np.nanmean(Pattern_sum_atm_component_fraction_CMIPS_COLD_change_array,axis=1)   
Pattern_sum_atm_component_fraction_CMIPS_COLD_change_mean=np.nanmean(Pattern_sum_atm_component_fraction_CMIPS_COLD_change_zonal,axis=1)   

##############
Patterns_relative_D0_CMIPS_COLD_change_list=[]
for i in range (n):

    test=Patterns_relative_D0_change_array[i,78:82,180:250]
    Patterns_relative_D0_CMIPS_COLD_change_list.append(test)  
Patterns_relative_D0_CMIPS_COLD_change_array=np.array(Patterns_relative_D0_CMIPS_COLD_change_list)
    
Patterns_relative_D0_CMIPS_COLD_change_zonal=np.nanmean(Patterns_relative_D0_CMIPS_COLD_change_array,axis=1)   
Patterns_relative_D0_CMIPS_COLD_change_mean=np.nanmean(Patterns_relative_D0_CMIPS_COLD_change_zonal,axis=1)   

###################



oceanic_flux_deviation_fraction_CMIPS_COLD_change_list=[]
for i in range (n):

    test=oceanic_flux_deviation_array_fraction[i,78:82,180:250]
    oceanic_flux_deviation_fraction_CMIPS_COLD_change_list.append(test)  
oceanic_flux_deviation_fraction_CMIPS_COLD_change_array=np.array(oceanic_flux_deviation_fraction_CMIPS_COLD_change_list)
    
oceanic_flux_deviation_fraction_CMIPS_COLD_change_zonal=np.nanmean(oceanic_flux_deviation_fraction_CMIPS_COLD_change_array,axis=1)   
oceanic_flux_deviation_fraction_CMIPS_COLD_change_mean=np.nanmean(oceanic_flux_deviation_fraction_CMIPS_COLD_change_zonal,axis=1)



Spatial_heterogeneity_fraction_array_CMIPS_COLD_change_list=[]
for i in range (n):

    test=Spatial_heterogeneity_fraction_array[i,78:82,180:250]
    Spatial_heterogeneity_fraction_array_CMIPS_COLD_change_list.append(test)  
Spatial_heterogeneity_fraction_array_CMIPS_COLD_change_array=np.array(Spatial_heterogeneity_fraction_array_CMIPS_COLD_change_list)
    
Spatial_heterogeneity_fraction_array_CMIPS_COLD_change_zonal=np.nanmean(Spatial_heterogeneity_fraction_array_CMIPS_COLD_change_array,axis=1)   
Spatial_heterogeneity_fraction_array_CMIPS_COLD_change_mean=np.nanmean(Spatial_heterogeneity_fraction_array_CMIPS_COLD_change_zonal,axis=1)   



###########################"



relative_sum_oceanic_component_fraction_CMIPS_COLD_change_list=[]
for i in range (n):

    test=(oceanic_somme_component_change[i,78:82,180:250])
    relative_sum_oceanic_component_fraction_CMIPS_COLD_change_list.append(test)  
relative_sum_oceanic_component_fraction_CMIPS_COLD_change_array=np.array(relative_sum_oceanic_component_fraction_CMIPS_COLD_change_list)
    
relative_sum_oceanic_component_fraction_CMIPS_COLD_change_zonal=np.nanmean(relative_sum_oceanic_component_fraction_CMIPS_COLD_change_array,axis=1)   
relative_sum_oceanic_component_fraction_CMIPS_COLD_change_mean=np.nanmean(relative_sum_oceanic_component_fraction_CMIPS_COLD_change_zonal,axis=1)   





RSST_somme_component_CMIPS_COLD_change_list=[]
for i in range (n):

    test=RSST_somme_component_change_ALPHA_MMM[i,78:82,180:250]
    RSST_somme_component_CMIPS_COLD_change_list.append(test)
    RSST_somme_component_CMIPS_COLD_change=np.array(RSST_somme_component_CMIPS_COLD_change_list)
RSST_somme_component_CMIPS_COLD_change_zonal=np.nanmean(RSST_somme_component_CMIPS_COLD_change,axis=1)   
RSST_somme_component_CMIPS_COLD_change_mean=np.nanmean(RSST_somme_component_CMIPS_COLD_change_zonal,axis=1)   













# #################################################################################################



relative_tauu_CMIPS_EQUATEUR_change_list=[]
for i in range (n):

    test=relative_tauu_change_array[i,78:82,130:270]
    relative_tauu_CMIPS_EQUATEUR_change_list.append(test)  
relative_tauu_CMIPS_EQUATEUR_change_array=np.array(relative_tauu_CMIPS_EQUATEUR_change_list)
    
relative_tauu_CMIPS_EQUATEUR_change_zonal=np.nanmean(relative_tauu_CMIPS_EQUATEUR_change_array,axis=1)   
relative_tauu_CMIPS_EQUATEUR_change_mean=np.nanmean(relative_tauu_CMIPS_EQUATEUR_change_zonal,axis=1)   



tos_biais_CMIPS_WEST_change_list=[]
for i in range (n):

    test=tos_biais_array[i,78:82,150:200]
    tos_biais_CMIPS_WEST_change_list.append(test)  
tos_biais_CMIPS_WEST_change_array=np.array(tos_biais_CMIPS_WEST_change_list)
    
tos_biais_CMIPS_WEST_change_zonal=np.nanmean(tos_biais_CMIPS_WEST_change_array,axis=1)   
tos_biais_CMIPS_WEST_change_mean=np.nanmean(tos_biais_CMIPS_WEST_change_zonal,axis=1)   





relative_tauu_CMIPS_WEST_change_list=[]
for i in range (n):

    test=relative_tauu_change_array[i,78:82,150:200]
    relative_tauu_CMIPS_WEST_change_list.append(test)  
relative_tauu_CMIPS_WEST_change_array=np.array(relative_tauu_CMIPS_WEST_change_list)
    
relative_tauu_CMIPS_WEST_change_zonal=np.nanmean(relative_tauu_CMIPS_WEST_change_array,axis=1)   
relative_tauu_CMIPS_WEST_change_mean=np.nanmean(relative_tauu_CMIPS_WEST_change_zonal,axis=1)   


tos_biais_CMIPS_EAST_change_list=[]
for i in range (n):

    test=tos_biais_array[i,78:82,220:260]
    tos_biais_CMIPS_EAST_change_list.append(test)  
tos_biais_CMIPS_EAST_change_array=np.array(tos_biais_CMIPS_EAST_change_list)
    
tos_biais_CMIPS_EAST_change_zonal=np.nanmean(tos_biais_CMIPS_EAST_change_array,axis=1)   
tos_biais_CMIPS_EAST_change_mean=np.nanmean(tos_biais_CMIPS_EAST_change_zonal,axis=1)   




relative_tauu_CMIPS_EAST_change_list=[]
for i in range (n):

    test=relative_tauu_change_array[i,78:82,220:260]
    relative_tauu_CMIPS_EAST_change_list.append(test)  
relative_tauu_CMIPS_EAST_change_array=np.array(relative_tauu_CMIPS_EAST_change_list)
    
relative_tauu_CMIPS_EAST_change_zonal=np.nanmean(relative_tauu_CMIPS_EAST_change_array,axis=1)   
relative_tauu_CMIPS_EAST_change_mean=np.nanmean(relative_tauu_CMIPS_EAST_change_zonal,axis=1)   





tos_biais_CMIPS_EQUATOR_change_list=[]
for i in range (n):

    test=tos_biais_array[i,78:82,130:270]
    tos_biais_CMIPS_EQUATOR_change_list.append(test)  
tos_biais_CMIPS_EQUATOR_change_array=np.array(tos_biais_CMIPS_EQUATOR_change_list)
    
tos_biais_CMIPS_EQUATOR_change_zonal=np.nanmean(tos_biais_CMIPS_EQUATOR_change_array,axis=1)   
tos_biais_CMIPS_EQUATOR_change_mean=np.nanmean(tos_biais_CMIPS_EQUATOR_change_zonal,axis=1)   




########################### curl 

# Calcul du curl du vent pour les régions spécifiques (WEST)
tauu_curl_CMIPS_WEST_change_list = []
for i in range(n):
    test = tauu_curl_change_array[i, 78:82, 150:200]
    tauu_curl_CMIPS_WEST_change_list.append(test)  
tauu_curl_CMIPS_WEST_change_array = np.array(tauu_curl_CMIPS_WEST_change_list)
    
tauu_curl_CMIPS_WEST_change_zonal = np.nanmean(tauu_curl_CMIPS_WEST_change_array, axis=1)   
tauu_curl_CMIPS_WEST_change_mean = np.nanmean(tauu_curl_CMIPS_WEST_change_zonal, axis=1)   

# Région EAST
tauu_curl_CMIPS_EAST_change_list = []
for i in range(n):
    test = tauu_curl_change_array[i, 78:82, 220:270]
    tauu_curl_CMIPS_EAST_change_list.append(test)  
tauu_curl_CMIPS_EAST_change_array = np.array(tauu_curl_CMIPS_EAST_change_list)
    
tauu_curl_CMIPS_EAST_change_zonal = np.nanmean(tauu_curl_CMIPS_EAST_change_array, axis=1)   
tauu_curl_CMIPS_EAST_change_mean = np.nanmean(tauu_curl_CMIPS_EAST_change_zonal, axis=1)

# Région EQUATOR (si nécessaire)
tauu_curl_CMIPS_EQUATOR_change_list = []
for i in range(n):
    test = tauu_curl_change_array[i, 78:82, 150:270]
    tauu_curl_CMIPS_EQUATOR_change_list.append(test)  
tauu_curl_CMIPS_EQUATOR_change_array = np.array(tauu_curl_CMIPS_EQUATOR_change_list)
    
tauu_curl_CMIPS_EQUATOR_change_zonal = np.nanmean(tauu_curl_CMIPS_EQUATOR_change_array, axis=1)   
tauu_curl_CMIPS_EQUATOR_change_mean = np.nanmean(tauu_curl_CMIPS_EQUATOR_change_zonal, axis=1)



# Calcul du curl du vent pour les régions spécifiques (WEST)
tauu_curl_CMIPS_WEST_change_list = []
for i in range(n):
    test = tauu_curl_change_array[i, 75:85, 150:200]
    tauu_curl_CMIPS_WEST_change_list.append(test)  
tauu_curl_CMIPS_WEST_change_array = np.array(tauu_curl_CMIPS_WEST_change_list)
    
tauu_curl_CMIPS_WEST_change_zonal = np.nanmean(tauu_curl_CMIPS_WEST_change_array, axis=1)   
tauu_curl_CMIPS_WEST_change_mean = np.nanmean(tauu_curl_CMIPS_WEST_change_zonal, axis=1)   

# Région EAST
tauu_curl_CMIPS_EAST_change_list = []
for i in range(n):
    test = tauu_curl_change_array[i, 75:85, 220:270]
    tauu_curl_CMIPS_EAST_change_list.append(test)  
tauu_curl_CMIPS_EAST_change_array = np.array(tauu_curl_CMIPS_EAST_change_list)
    
tauu_curl_CMIPS_EAST_change_zonal = np.nanmean(tauu_curl_CMIPS_EAST_change_array, axis=1)   
tauu_curl_CMIPS_EAST_change_mean = np.nanmean(tauu_curl_CMIPS_EAST_change_zonal, axis=1)

# Région EQUATOR (si nécessaire)
tauu_curl_CMIPS_EQUATOR_change_list = []
for i in range(n):
    test = tauu_curl_change_array[i, 75:85, 150:270]
    tauu_curl_CMIPS_EQUATOR_change_list.append(test)  
tauu_curl_CMIPS_EQUATOR_change_array = np.array(tauu_curl_CMIPS_EQUATOR_change_list)
    
tauu_curl_CMIPS_EQUATOR_change_zonal = np.nanmean(tauu_curl_CMIPS_EQUATOR_change_array, axis=1)   
tauu_curl_CMIPS_EQUATOR_change_mean = np.nanmean(tauu_curl_CMIPS_EQUATOR_change_zonal, axis=1)


