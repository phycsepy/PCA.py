#!/usr/bin/pytho3
# Sometimes the data may have too many variables such that most of the variables are correlated
# Model on the whole data => poor accuracy
# Soln- PCA
# Working principle of PCA - dimensionality reduction
# PCA - method of extracting important variables with a motive to capture as much info as possible
#     - extracts low dimensional set of variables from a high dimensional set
# Here we are not going to predict y
# The PCA unsupervised learning approach tries to learn the strength of relationship of varaiables
import pandas as pd 
df = pd.DataFrame(columns=['Calory','Breakfast','Lunch','Dinner','Excercise','Body Shape'])
df.loc[0]=[1200,1,0,0,2,'Skinny']
df.loc[1]=[2800,1,1,1,1,'Fat']
df.loc[2]=[3500,2,2,1,0,'Skinny']
df.loc[3]=[1400,0,1,0,3,'Skinny']
df.loc[4]=[1600,1,0,2,0,'Normal']
df.loc[5]=[3200,1,2,1,1,'Fat']
df.loc[6]=[1750,1,0,0,1,'Skinny']
df.loc[7]=[1600,1,0,0,0,'Skinny']
print(df)
# Split feature vectors and labels
x=df[['Calory','Breakfast','Lunch','Dinner','Excercise']]
y=df[['Body Shape']]
print(y)
# The mean of 'Calory' will be very high compared to the means of the other 4 columns....so we have to normalise it 
# Using StandardScaler (-1,1)

from sklearn.preprocessing import StandardScaler as ss
x_std=ss().fit_transform(x)
print(x_std)
# Covariance matrix of features

# Features are columns from x_std

import numpy as np
features=x_std.T
covariance_matrix=np.cov(features)
print(covariance_matrix)


#PCA believed that the points which have close cov or corr have more impact
#Eigen vector- the respected features....not suppressed

# .t means transform

# Eigen vectors and eigen values from co-variance matrix 
#The eigenvector with the largest eigenvalue is the direction along which the data set has the maximum variance.

eig_vals , eig_vecs = np.linalg.eig(covariance_matrix)
print(' Eigen values : ',eig_vals,sep='\n' )
# Reduce dimension to 1 dimension
eig_vals[0]/sum(eig_vals)   #...... = 1st principal/performing component
# Project datapoint onto selected eigen vector
projected_x=x_std.dot(eig_vecs.T[0])

#...The columns in the dataset has been converted into 1 influential column....represented on x-axis
# since its unsupervised learning, the value of the y-axis is 0
print(projected_x) 
result=pd.DataFrame(projected_x,columns=['PC1'])   # PC1=Principal Component 1
result['y-axis']=0.0
result['label']=y
print(result)
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
sns.lmplot('PC1','y-axis',data=result,fit_reg=False,scatter_kws={'s':50},hue='label')

#title
plt.title('PCA result')

# hue='label' .....unknown labels
# fit_reg.... fit regression
# In the above plot, the left side is for breakfast and lunch
# the right side is for dinner and exercise

# Even if you have low calorie intake, if u hv breakfast or lunch then the body shape will be FAT
# To be NORMAL, u should hv moderate values for all the independent variables in the original dataset
#...since NORMAL is in the center
# Even if you have high calorie intake, if u hv dinner and excercise then the body shape will be SKINNY

df.loc[4]=[1600,1,0,10,0,'Normal']
print(df)
x=df[['Calory','Breakfast','Lunch','Dinner','Excercise']]
y=df[['Body Shape']]
x_std=ss().fit_transform(x)
x_std
features=x_std.T
covariance_matrix=np.cov(features)
print(covariance_matrix)
eig_vals,eig_vecs=np.linalg.eig(covariance_matrix)
print(' Eigen values : ',eig_vals,sep='\n' )
    # Project datapoint onto selected eigen vector
projected_x=x_std.dot(eig_vecs.T[0])
projected_x
result=pd.DataFrame(projected_x,columns=['PC1'])   # PC1=Principal Component 1
result['y-axis']=0.0
result['label']=y
result
sns.lmplot('PC1','y-axis',data=result,fit_reg=False,scatter_kws={'s':50},hue='label')

#title
plt.title('PCA result')

# Here the point for NORMAL has shifted to the right since the value for dinner has increased
df.loc[4]=[1600,1,0,2,10,'Normal']
df
x=df[['Calory','Breakfast','Lunch','Dinner','Excercise']]
y=df[['Body Shape']]
x_std=ss().fit_transform(x)
x_std
features=x_std.T
covariance_matrix=np.cov(features)
print(covariance_matrix)
eig_vals,eig_vecs=np.linalg.eig(covariance_matrix)
print('Eigen values :',eig_vals,sep='\n')
eig_vals[0]/sum(eig_vals)   #...... = 1st principal/performing component
# Project datapoint onto selected eigen vector
projected_x=x_std.dot(eig_vecs.T[0])
projected_x
result=pd.DataFrame(projected_x,columns=['PC1'])   # PC1=Principal Component 1
result['y-axis']=0.0
result['label']=y
result
sns.lmplot('PC1','y-axis',data=result,fit_reg=False,scatter_kws={'s':50},hue='label')

#title
plt.title('PCA result')
f.loc[4]=[1600,10,0,2,0,'Normal']
x=df[['Calory','Breakfast','Lunch','Dinner','Excercise']]
y=df[['Body Shape']]
x_std=StandardScaler().fit_transform(x)
features=x_std.T
covariance_matrix=np.cov(features)
eig_vals,eig_vecs=np.linalg.eig(covariance_matrix)
print('\nEigen values \n%s' %eig_vals)
eig_vals[0]/sum(eig_vals)   #...... = 1st principal/performing component
projected_x=x_std.dot(eig_vecs.T[0])
result=pd.DataFrame(projected_x,columns=['PC1'])   # PC1=Principal Component 1
result['y-axis']=0.0
result['label']=y
result
sns.lmplot('PC1','y-axis',data=result,fit_reg=False,scatter_kws={'s':50},hue='label')

#title
plt.title('PCA result')

# Here the point for NORMAL has shifted to the leftmost side since the value for Breakfast has increased