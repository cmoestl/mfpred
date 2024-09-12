#!/usr/bin/env python
# coding: utf-8

# ### Real time deployment of the machine learning algorithms for predicting the magnetic flux rope structure in coronal mass ejections
# 
# This is a code adapted for real time Bz prediction from the Reiss et al. 2021 Space Weather paper.
# https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021SW002859
# 
# This notebook is used for real time deployment.
# 
# 
# ### Update
# last update 2024 Sep 11.
# 
# ### Ideas
# 
# - deploy in real time for data files for STEREO-A and NOAA RTSW  under folder data_path:
# "stereoa_beacon_gsm_last_35days_now.p" and "noaa_rtsw_last_35files_now.p"
# - read in ML model trained with the notebooks mfrpred_real_bz (done), mfpred_real_btot (need to update)
# 
# - continous deployment, look at results during CMEs
# - assess progression of results in real time as more of the CME is seen
# - needs different trained model for each timestep, i.e. for different hours after sheath and MFR entry?
# 
# - add general Bz distribution plots here at the end
# 
# 
# ### Future
# - forecast of the cumulative southward Bz during a geomagnetic storm?
# - start at time of shock, and then decrease the error bars with time
# - correlate with Dst
# 
# pattern recognition
# https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2016SW001589
# 
# bz after shocks
# https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2018SW002056
# 
# #### Authors: 
# M.A. Reiss (1), C. Möstl (2), R.L. Bailey (3), and U. Amerstorfer (2), Emma Davies (2), Eva Weiler (2)
# 
# (1) NASA CCMC, 
# (2) Austrian Space Weather Office, GeoSphere Austria
# (3) Conrad Observatory, GeoSphere Austria
# 

# In[118]:


########### controls

print()
print('started mfrpred_deploy.py')




################

import time

#test execution times
t0all = time.time()


import os
import sys
import copy
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, num2date
from matplotlib import cm
import numpy as np
import pickle
from scipy import stats
import scipy.io
import time
import datetime

# Visualisation
import sunpy.time
import seaborn as sns
import pandas as pd
from pandas.plotting import scatter_matrix
from sunpy.time import parse_time

# Machine learning
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.linear_model import ElasticNet, HuberRegressor, Lars, LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor, RANSACRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Don't print warnings
import warnings
warnings.filterwarnings('ignore')

os.system('jupyter nbconvert --to script mfrpred_deploy.ipynb')    

#get data paths
if sys.platform == 'linux': 
    
    from config_server import data_path
    from config_server import noaa_path
    from config_server import wind_path
    from config_server import stereoa_path
    from config_server import data_path_ml
    
if sys.platform =='darwin':  

    from config_local import data_path
    from config_local import noaa_path
    from config_local import wind_path
    from config_local import stereoa_path
    from config_local import data_path_ml



# ## load real time data

# In[110]:


filenoaa='noaa_rtsw_last_35files_now.p'
[noaa,hnoaa]=pickle.load(open(data_path+filenoaa, "rb" ) ) 

file_sta_beacon_gsm='stereoa_beacon_gsm_last_35days_now.p'  
[sta,hsta]=pickle.load(open(data_path+file_sta_beacon_gsm, "rb" ) )  

print('real time NOAA RTSW and STEREO-A data loaded')

#cutout last 10 days
start=datetime.datetime.utcnow() - datetime.timedelta(days=10)
end=datetime.datetime.utcnow() 

ind=np.where(noaa.time > start)[0][0]
noaa=noaa[ind:]

ind2=np.where(sta.time > start)[0][0]
sta=sta[ind2:]


# In[111]:


###plot NOAA
plt.figure(1,figsize=(12, 4))
plt.plot(noaa.time,noaa.bz, '-b',lw=0.5)
plt.plot(noaa.time,noaa.bt,'-k')

plt.title("NOAA RTSW")  # Adding a title
plt.xlabel("time")  # Adding X axis label
plt.ylabel("B [nT]")  # Adding Y axis label
plt.grid(True)  # Adding a grid

plt.xlim(start, end)

#plot STEREO-A

plt.figure(2,figsize=(12, 4))
plt.plot(sta.time,sta.bz, '-b',lw=0.5)
plt.plot(sta.time,sta.bt,'-k')

plt.title("STEREO-A beacon")  # Adding a title
plt.xlabel("time")  # Adding X axis label
plt.ylabel("B [nT]")  # Adding Y axis label
plt.grid(True)  # Adding a grid

plt.xlim(start, end)


# In[112]:


sns.set_context("talk")     
sns.set_style('whitegrid')

fig=plt.figure(figsize=(12,6),dpi=100)
#ax1 = plt.subplot(111) 


#cutout last 10 hours, e.g. sheath is over and flux rope starts
start=datetime.datetime.utcnow() - datetime.timedelta(hours=10)
end=datetime.datetime.utcnow() 

ind=np.where(noaa.time > start)[0][0]
noaa_cut=noaa[ind:]

ind2=np.where(sta.time > start)[0][0]
sta_cut=sta[ind2:]

###plot NOAA
plt.figure(1,figsize=(12, 4))
plt.plot(noaa_cut.time,noaa_cut.bz, '-b',lw=0.5)
plt.plot(noaa_cut.time,noaa_cut.bt,'-k')

plt.title("NOAA RTSW")  # Adding a title
plt.xlabel("time")  # Adding X axis label
plt.ylabel("B [nT]")  # Adding Y axis label
plt.grid(True)  # Adding a grid

plt.xlim(start, end)

#plot STEREO-A

plt.figure(2,figsize=(12, 4))
plt.plot(sta_cut.time,sta_cut.bz, '-b',lw=0.5)
plt.plot(sta_cut.time,sta_cut.bt,'-k')

plt.title("STEREO-A beacon")  # Adding a title
plt.xlabel("time")  # Adding X axis label
plt.ylabel("B [nT]")  # Adding Y axis label
plt.grid(True)  # Adding a grid

plt.xlim(start, end)



# ### load ML model

# In[113]:


#what the model numbers mean
#model1 = models['lr'] 
#model2 = models['rfr'] 
#model3 = models['gbr'] 

feature_hours=10
[model1,model2,model3]=pickle.load(open('trained_models/bz_'+str(feature_hours)+'h_model.p','rb'))


print()
print('ML model loaded')

#model1.predict()
model2


#y_pred1 = model1.predict(X_test)

#y_pred1 sind die Bz predictions



# ### Apply ML model
# 

# In[114]:


## how to apply, first calculate features from current data? and then put into model

#feature space - map to model, get output

print()
print('ML model to be run on real time data')


# ### Make output data files and plots

# In[ ]:


print()
print('results')
print()


# In[ ]:





# ### General Bz overview plots

# In[115]:


##load ICME catalog

[ic,header,parameters] = pickle.load(open('data/ICMECAT/HELIO4CAST_ICMECAT_v22_pandas.p', "rb" ))

print()
print('ICMECAT loaded')

# Spacecraft
isc = ic.loc[:,'sc_insitu'] 

# Shock arrival or density enhancement time
icme_start_time = ic.loc[:,'icme_start_time']
icme_start_time_num = date2num(np.array(icme_start_time))

# Start time of the magnetic obstacle (mo)
mo_start_time = ic.loc[:,'mo_start_time']
mo_start_time_num = date2num(np.array(mo_start_time))

# End time of the magnetic obstacle (mo)
mo_end_time = ic.loc[:,'mo_end_time']
mo_end_time_num = date2num(np.array(mo_end_time))

#get indices for each target
wini=np.where(ic.sc_insitu=='Wind')[0]
stai=np.where(ic.sc_insitu=='STEREO-A')[0]
stbi=np.where(ic.sc_insitu=='STEREO-B')[0]
pspi=np.where(ic.sc_insitu=='PSP')[0]
soloi=np.where(ic.sc_insitu=='SolarOrbiter')[0]
bepii=np.where(ic.sc_insitu=='BepiColombo')[0]
ulyi=np.where(ic.sc_insitu=='Ulysses')[0]
messi=np.where(ic.sc_insitu=='Messenger')[0]
vexi=np.where(ic.sc_insitu=='VEX')[0]


# In[116]:


ic.keys()


# In[117]:


##plot for minimum Bz vs time

sns.set_context("talk")     
sns.set_style('whitegrid')

fig=plt.figure(figsize=(12,6),dpi=100)
ax1 = plt.subplot(111) 

ax1.plot(ic['icme_start_time'][wini],ic['mo_bzmin'][wini],'og',markersize=3,label='Wind ICME min(Bz)')
ax1.plot(ic['icme_start_time'][stai],ic['mo_bzmin'][stai],'or',markersize=3,label='STEREO-A ICME min(Bz)')
ax1.plot(ic['icme_start_time'][stbi],ic['mo_bzmin'][stbi],'ob',markersize=3,label='STEREO-B ICME min(Bz)')
ax1.set_ylim(-100,30)
plt.legend(fontsize=10,loc=2)
plt.tight_layout()
plt.title('Bz in ICME magnetic obstacles for Wind, STEREO-A/B  ICMECAT mo_bzmin')
plt.savefig('plots/icme_bz_time.png')


print()
print('saved plots/icme_bz_time.png')
print()
print()



# In[ ]:





# In[ ]:





# In[ ]:


t1all = time.time()

print(' ')
print(' ')
print(' ')
print('------------------')
print('Runtime for full high frequency data update:', np.round((t1all-t0all)/60,2), 'minutes')
print('------------------')

