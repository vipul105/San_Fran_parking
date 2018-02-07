# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 19:51:00 2018

@author: Vipul Satone
"""

### double nimfa


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as math
import nimfa
from sklearn.cluster import KMeans
import collections, numpy
from pydoc import help
from scipy.stats.stats import pearsonr

'''
data3 = pd.read_csv('G:\\uiuc\\igl data\\ews\\occupancy/jan_occ_2013.csv', index_col = 0)
data2 = data3.iloc[:,4:].fillna(0)
'''
W = pd.read_csv('G:\\uiuc\\igl data\\ews\\occ bigmatrix analysis\\W_20_Beta_0.0_.csv', index_col = 0)
H = pd.read_csv('G:\\uiuc\\igl data\\ews\\occ bigmatrix analysis\\H_20_Beta_0.0_.csv', index_col = 0)
data2 = pd.read_csv('G:\\uiuc\\igl data\\ews\\Corrected_big_matrix_occupancy.csv', index_col = 0)




global NONZEROS
NONZEROS = ~np.isnan(data2)
data = data2.fillna(0)
ss = lambda x: x/x.max()
#data = pd.DataFrame(data).apply(ss, axis =0)
D = data
global norm_D
norm_D = np.linalg.norm(D)
def m(A,B): return np.multiply(A,B)
def N(A): return m(A, NONZEROS)
def calculate_error(D, W, H):
    return np.linalg.norm(D - N(np.matmul(W,H)))/norm_D*100



'''
k = 5
np.random.seed(0)
w_r = np.random.random((2880, k))
h_r = np.random.random((k,296))
betaa = 0.0
snmf = nimfa.Snmf(np.matrix(data),rank=k, beta = betaa ,max_iter=1000,  W = w_r, H = h_r,version='r', min_residuals  = 0.0000001)
snmf_fit = snmf()
W = snmf_fit.fit.W
H = snmf_fit.fit.H
'''
error =  calculate_error(D, W, H)


time=[]
for i in range(24):
    for j in range(4):
        p = i*100 + j*15
        a = p
        time.append(a)



'''
W = pd.read_csv("G:\\uiuc\\igl data\\ews\\bas khatam kar\\01o\\W_5_Beta_0.0_.csv", index_col = 0)
H = pd.read_csv("G:\\uiuc\\igl data\\ews\\bas khatam kar\\01o\\H_5_Beta_0.0_.csv", index_col = 0)
'''

h1 = H.copy()

f = lambda x: x/x.max()
f0 = lambda x: (x*100)/x.max()
H = pd.DataFrame(H).apply(f, axis =1)
H = pd.DataFrame(H).apply(f0, axis =0)


# elbow curve
cluster_range = range( 1,70)
cluster_err = []

for num_center in cluster_range:
  cluster = KMeans( num_center )
  cluster.fit( H.T )
  cluster_err.append( cluster.inertia_ )
  
cluster_df = pd.DataFrame( { "num_center":cluster_range, "cluster_err": cluster_err } )
plt.figure(figsize=(8,4))
plt.plot( cluster_df.num_center, cluster_df.cluster_err, marker = "o" )
plt.show()


# we chose k = 10


# applying model
kmeanModel = KMeans(n_clusters=15).fit(H.T)
kmeanModel.fit(H.T)
center = kmeanModel.cluster_centers_
labels = kmeanModel.labels_


nam = list(data.columns.values)
labl =[]
labl = pd.DataFrame(labl)
labl['name'] = nam
labl['cluster'] = labels

data = np.matrix(data)
df_0 = data.T[labels == 0,:].T
cluster_0 = labl[labl['cluster'] == 0].name
df_0 = (pd.DataFrame(df_0.T).set_index(cluster_0)).T

df_1 = data.T[labels == 1,:].T
cluster_1 = labl[labl['cluster'] == 1].name
df_1 = (pd.DataFrame(df_1.T).set_index(cluster_1)).T

df_2 = data.T[labels == 2,:].T
cluster_2 = labl[labl['cluster'] == 2].name
df_2 = (pd.DataFrame(df_2.T).set_index(cluster_2)).T


df_3 = data.T[labels == 3,:].T
cluster_3 = labl[labl['cluster'] == 3].name
df_3 = (pd.DataFrame(df_3.T).set_index(cluster_3)).T


df_4 = data.T[labels == 4,:].T
cluster_4 = labl[labl['cluster'] == 4].name
df_4 = (pd.DataFrame(df_4.T).set_index(cluster_4)).T


df_5 = data.T[labels == 5,:].T
cluster_5 = labl[labl['cluster'] == 5].name
df_5 = (pd.DataFrame(df_5.T).set_index(cluster_5)).T


df_6 = data.T[labels == 6,:].T
cluster_6 = labl[labl['cluster'] == 6].name
df_6 = (pd.DataFrame(df_6.T).set_index(cluster_6)).T


df_7 = data.T[labels == 7,:].T
cluster_7 = labl[labl['cluster'] == 7].name
df_7 = (pd.DataFrame(df_7.T).set_index(cluster_7)).T


df_8 = data.T[labels == 8,:].T
cluster_8 = labl[labl['cluster'] == 8].name
df_8 = (pd.DataFrame(df_8.T).set_index(cluster_8)).T


df_9 = data.T[labels == 9,:].T
cluster_9 = labl[labl['cluster'] == 9].name
df_9 = (pd.DataFrame(df_9.T).set_index(cluster_9)).T

data = np.matrix(data)
df_10 = data.T[labels == 10,:].T
cluster_10 = labl[labl['cluster'] == 10].name
df_10 = (pd.DataFrame(df_10.T).set_index(cluster_10)).T

df_11 = data.T[labels == 11,:].T
cluster_11 = labl[labl['cluster'] == 11].name
df_11 = (pd.DataFrame(df_11.T).set_index(cluster_11)).T

df_12 = data.T[labels == 12,:].T
cluster_12 = labl[labl['cluster'] == 12].name
df_12 = (pd.DataFrame(df_12.T).set_index(cluster_12)).T


df_13 = data.T[labels == 13,:].T
cluster_13 = labl[labl['cluster'] == 13].name
df_13 = (pd.DataFrame(df_13.T).set_index(cluster_13)).T


df_14 = data.T[labels == 14,:].T
cluster_14 = labl[labl['cluster'] == 14].name
df_14 = (pd.DataFrame(df_14.T).set_index(cluster_14)).T



# cluster 0
def cons_sig(df):
    kk = 1
    np.random.seed(0)
    w_r = np.random.random((df.shape[0], kk))
    h_r = np.random.random((kk,df.shape[1]))
    betaa = 0.0
    snmf0 = nimfa.Snmf(np.matrix(df),rank=kk, beta = betaa ,max_iter=1000,  W = w_r, H = h_r,version='r', min_residuals  = 0.0000001)
    snmf0_fit = snmf0()
    W00 = snmf0_fit.fit.W
    H0 = snmf0_fit.fit.H
    return W00
'''

def plot_cluster(W0, df,a):
    fig0 = plt.figure()
    for i in range(10):    
        plt.plot(time, W0[(0+(i*96)):(96+(i*96)),0])
        plt.ylabel('numbers of cars entering')
        plt.xlabel('time')
        plt.title('Visualization of signature in cluster')
    plt.show()
    ax = 'G:\\uiuc\\igl data\\ews\\bas khatam kar\\double nimfa\\signature_' + str(a) + '_.png'
    fig0.savefig(ax)  


    fig1 = plt.figure()
    for i in range(5):    
        plt.plot(time, df.iloc[(96*1):(96*2),i+2])
        plt.ylabel('numbers of cars entering')
        plt.xlabel('time')
        plt.title('visualization of parking lots behaviour in cluster')
    plt.show()
    ax = 'G:\\uiuc\\igl data\\ews\\bas khatam kar\\double nimfa\\parking_lots_' + str(a) + '_.png'
    fig1.savefig(ax)  
    '''


def plot_cluster(W0, df,a):
    fig0 = plt.figure()
    plt.subplot(2, 1, 1)
    for i in range(10):    
        plt.plot(time, W0[(0+(i*96)):(96+(i*96)),0])
        plt.ylabel('numbers of cars entering')
        plt.xlabel('time')
        title = 'Visualization of signature in cluster ' + str(a)
        plt.title(title)
    #plt.show()
    #ax = 'G:\\uiuc\\igl data\\ews\\bas khatam kar\\double nimfa\\signature_' + str(a) + '_.png'
    #fig0.savefig(ax)  

    plt.subplot(2, 1, 2)
    for i in range(5):    
        plt.plot(time, df.iloc[(96*1):(96*2),i+2])
        plt.ylabel('numbers of cars entering')
        plt.xlabel('time')
        title1 = 'visualization of parking lots behaviour in cluster ' + str(a)
        plt.title(title1)
    plt.show()
    ax = 'G:\\uiuc\\igl data\\ews\\bas khatam kar\\double nimfa\\parking_lots_' + str(a) + '_.png'
    fig0.savefig(ax) 




dict_w = {}
for r in range(15):    
    dict_w[r] = cons_sig(eval('df_' + str(r) ))

plot_cluster(dict_w[0], df_0,0)
plot_cluster(dict_w[1], df_1,1)
plot_cluster(dict_w[2], df_2,2)
plot_cluster(dict_w[3], df_3,3)
plot_cluster(dict_w[4], df_4,4)
plot_cluster(dict_w[5], df_5,5)
plot_cluster(dict_w[6], df_6,6)
plot_cluster(dict_w[7], df_7,7)
plot_cluster(dict_w[8], df_8,8)
plot_cluster(dict_w[9], df_9,9)
plot_cluster(dict_w[10], df_10,10)
plot_cluster(dict_w[11], df_11,11)
plot_cluster(dict_w[12], df_12,12)
plot_cluster(dict_w[13], df_13,13)
plot_cluster(dict_w[14], df_14,14)

# pearson coefficient
f = {}
for dd in range(15):
    
    d = 'df_' + str(dd) 
    p = 'per_' + str(dd)

    pp = np.zeros((eval(d).shape[1],eval(d).shape[1]))
    for i in range(eval(d).shape[1]):
        for j in range(eval(d).shape[1]):
            pp[i,j] = pearsonr(eval(d+ '.iloc[:,i]'), eval(d+ '.iloc[:,j]'))[0]
            f[dd] = pp

# saving corelation matrix
for i in range(15):    
    ax = 'G:\\uiuc\\igl data\\ews\\bas khatam kar\\double nimfa\\corr_mat_cluster_' + str(i) + '.csv'
    pd.DataFrame(f[i]).to_csv(ax, sep =',')

# saving clusters 
for i in range(15):    
    ax = 'G:\\uiuc\\igl data\\ews\\bas khatam kar\\double nimfa\\df_' + str(i) + '.csv'
    p = 'df_' + str(i)
    pd.DataFrame(eval(p)).to_csv(ax, sep =',')

# saving corelation matrix
for i in range(15):    
    ax = 'G:\\uiuc\\igl data\\ews\\bas khatam kar\\double nimfa\\signatures_' + str(i) + '.csv'
    pd.DataFrame(dict_w[i]).to_csv(ax, sep =',')

# saving cluster names
for i in range(15):    
    ax = 'G:\\uiuc\\igl data\\ews\\bas khatam kar\\double nimfa\\cluster_' + str(i) + '.csv'
    p = 'cluster_' + str(i)
    pd.DataFrame(eval(p)).to_csv(ax, sep =',')











