#!/usr/bin/env python
# coding: utf-8

# # Group 3 Members
# 
# 1. AKANDE Oluwatosin Adetoye
# 2. Ange Clement Akazan
# 3. Jeanne NIYONTEZE
# 4. Redempta Blandine ISHIME
# 5. Enock MWIZERWA
# 6. Mahamat Azibert ABDELWAHAB

# In[230]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import  StandardScaler
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as shc
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


# # Exploring the dataset lagos.csv

# In[231]:


df = pd.read_csv('C:/Users/Akande/Desktop/RMCS/Lagos.csv')
df = df.drop('Unnamed: 0', axis=1)
df.head()


# In[232]:


X = df.iloc[:, 1:]
X.head()


# In[233]:


# correlation matrix
cormat = X.corr()
round(cormat,2)


# In[234]:


plt.figure(figsize=(8,6))
sns.heatmap(cormat);


# # Scalling data (Standardize)

# In[235]:


# Scale the Data 
scaled = scale(X) 

# Scaled data as DataFrame 
scaled_df = pd.DataFrame(scaled) 
scaled_df.columns = X.columns

scaled_df.head()


# # Clustering

# ## 1. Single linkage

# In[236]:


# Single linkage
linked = shc.linkage(scaled_df.T, 'single')

plt.figure(figsize=(10, 6))
shc.dendrogram(linked, labels=scaled_df.columns)

plt.title('Single linkage')
plt.show()


# ## 2. Average Linkage

# In[237]:


# Average linkage
linked = shc.linkage(scaled_df.T, 'average')

plt.figure(figsize=(10, 7))
shc.dendrogram(linked, labels=scaled_df.columns)

plt.title('Average linkage')
plt.show()


# ## 3. Wards Algorithm

# In[238]:


# ward linkage
linked = shc.linkage(scaled_df.T, 'ward')

plt.figure(figsize=(10, 7))
shc.dendrogram(linked, labels=scaled_df.columns)

plt.title('Wards Algorithm')
plt.show()


# # 2. Principal Component Analysis

# In[239]:


# PCA (Unrotated)
pca = PCA() 
new_data = pca.fit_transform(scaled) 


# In[240]:


# New data as dataframe
cols = ['PF'+str(i) for i in range(1, X.shape[1]+1)]
new_df = pd.DataFrame(new_data, columns=cols)
new_df['Years'] = df['Year']
new_df.head()


# # New data

# In[241]:


# plotting new data on first two principal factors
plt.figure(figsize=(6,6))
sns.scatterplot(data = new_df, x="PF1", y="PF2", hue='Years')
plt.title('Scatter plot of new data')

plt.xlabel('PF1')
plt.ylabel('PF2')
plt.show()


# ## Unrotated PCA 
# 
# ### Loadings

# In[242]:


# loadings
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# display loadings as dataframe
loading_df = pd.DataFrame(loadings, columns=cols, index=X.columns)

loading_df


# ## Screeplot

# In[247]:


# screen plot
PC_values = np.arange(pca.n_components_) + 0
plt.plot(PC_values, pca.explained_variance_ratio_*10, 'o-', linewidth=2, color='blue')
plt.title('Screeplot')
plt.xlabel('Principal Component')
plt.ylabel('Egienvalule')
plt.show()


# ## Barplot explained variance 

# In[211]:


plt.figure(figsize=(6,5))
sns.barplot(x=loading_df.columns, y = pca.explained_variance_*10)
plt.title('Explained variance')
plt.ylabel('Percentage (%)')
plt.show()


# ## PCA Rotated

# In[223]:


data=pd.read_csv('C:/Users/Akande/Desktop/RMCS/Lagos.csv')
data=data.drop(['Unnamed: 0','Year'], axis=1)

scaler = StandardScaler()
scaled = scaler.fit_transform(data)

rotated = FactorAnalyzer(n_factors=2, rotation='varimax', method="principal", 
                    is_corr_matrix=False)
rotated.fit(scaled)


# In[224]:



rotatedloading = pd.DataFrame(rotated.loadings_, columns=['PF1', 'PF2'], index=data.columns)
a2, b2 ,c2= rotated.get_factor_variance()
row3=pd.DataFrame(a2,columns=['Expl.Var'],index=['PF1', 'PF2']).T
row4=pd.DataFrame(b2,columns=['Prp.Totl'],index=['PF1', 'PF2']).T
pca3 = pd.concat([rotatedloading,row3,row4], ignore_index=False)


# In[225]:


pca3.style.applymap(lambda x: 'color : red' if abs(x)>0.7and x<1 else '')


# In[226]:


scorer = pd.DataFrame(rotated.transform(scaled), columns=['PC1', 'PC2'],index=data.index)
scorer


# In[227]:


plt.style.use("ggplot")
plt.figure(figsize=(8,5))
plt.plot(fa.get_eigenvalues()[0], marker='o')
plt.xlabel("Eigenvalue number")
plt.ylabel("Eigenvalue size")
plt.title("Scree Plot")


# In[ ]:





# In[ ]:




