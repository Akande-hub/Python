#!/usr/bin/env python
# coding: utf-8

# # Group 3
# 
# 1. AKANDE Oluwatosin Adetoye    
# 2. Ange Clement Akazan             
# 3. Jeanne NIYONTEZE                  
# 4. Redempta Blandine ISHIME        
# 5. Enock MWIZERWA                    
# 6. Mahamat Azibert ABDELWAHAB 

# # Importing packages

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy import signal
import statsmodels.api as sm

import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


# # Loading dataset

# In[2]:


df = pd.read_csv('C:/Users/Akande/Desktop/RMCS/Lagos1.csv')
df.head()


# ### Extraction variables

# In[3]:


# extracting Year, precipitation and temperature
X = pd.DataFrame(df[['PRED', 'TMPD']].values, index = df['Year'], columns=['PRED', 'TMPD'])
X.head()


# ### Plotting
# 

# In[4]:


sns.lineplot(data = X['PRED'], err_style=None)
plt.title('The precipitation in Lagos')
plt.xlabel('Year')
plt.ylabel('Precipitation');


# In[5]:


sns.lineplot(data = X['TMPD'], err_style=None)
plt.title('The temperature in Lagos')
plt.xlabel('Year')
plt.ylabel('Temperature [°C]');


# ### Detrending variables

# In[6]:


#temperature
detrended_tmp = signal.detrend(X['TMPD'])
detrended_tmp_df = pd.DataFrame(detrended_tmp, index = X.index)
sns.lineplot(data = detrended_tmp_df, err_style=None, legend=None)
plt.title('Detrended temperature', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Temperature [°C]', fontsize=16)
plt.show()

#precipitation
detrended_prep = signal.detrend(X['PRED'])
detrended_prep_df = pd.DataFrame(detrended_prep, index = X.index)
sns.lineplot(data = detrended_prep_df, err_style=None, legend=None)
plt.title('Detrended precipitation', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Precipitation', fontsize=16)
plt.show()


# ### Auto - correlation

# In[7]:


# precipitation
plot_acf(detrended_prep_df, lags=30)
plt.show()


# In[8]:


#temperature
plot_acf(detrended_tmp_df, lags=30)
plt.show()


# # Cross correlation

# In[9]:


#calculate cross correlation
cc = sm.tsa.stattools.ccf(detrended_tmp_df, detrended_prep_df, adjusted=False)

plt.plot(cc);


# # Wavelet

# In[23]:


N = 10000
x_values_wvt = np.linspace(0, 100, N)
amplitudes = [4, 1.5, 9]
frequencies = [2, 5, 3]
y_values_0 = amplitudes[0]*np.cos(2*np.pi*frequencies[0]*detrended_prep_df) 
y_values_1 = amplitudes[1]*np.sin(2*np.pi*frequencies[1]*detrended_prep_df) 
y_values_2 = amplitudes[2]*np.cos(2*np.pi*frequencies[2]*detrended_prep_df + 0.4) 
full_signal_values = (y_values_0 + y_values_1 - y_values_2)

wavelet='cmor0.7-1.5'

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
fig.subplots_adjust(hspace=0.3)
ax1.plot(detrended_prep_df, prep, ax1.set_xlim(0, 5))
ax1.set_title('Example: time domain signal with linear combination of sin() and cos() waves')
ax2 = scg.cws(detrended_prep_df, full_signal_values, scales=np.arange(1, 150), wavelet=wavelet,
        ax=ax2, cmap="jet", cbar=None, ylabel="Period [seconds]", xlabel="Time [seconds]",
        title='Example: scaleogram with linear period yscale')


# In[ ]:




