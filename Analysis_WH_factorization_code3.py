

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as math
import nimfa
from sklearn.cluster import KMeans
import collections, numpy


data2 = pd.read_csv("U:/SF_data/Big_matrix_occupancy.csv", index_col = 0)

#data = data/ data.max(axis = 0)

global NONZEROS
NONZEROS = ~np.isnan(data2)
data = data2.fillna(0)
D = data
global norm_D
norm_D = np.linalg.norm(D)
def m(A,B): return np.multiply(A,B)
def N(A): return m(A, NONZEROS)
def calculate_error(D, W, H):
    return np.linalg.norm(D - N(np.matmul(W,H)))/norm_D*100
# for loop
bet = np.zeros([20,2])
spsh = np.zeros([20,2])
spsw = np.zeros([20,2])
rss = np.zeros([20,2])
evals = np.zeros([20,2])
kl = np.zeros([20,2])
H_min_mat = np.zeros([20,2])
W_min_mat = np.zeros([20,2])


for b in range(0,2):
    for t in range((20)):
        k = t+1
        print(k)
        np.random.seed(0)
        w_r = np.random.random((40320, k))
        h_r = np.random.random((k,285))
        betaa = (b)/10.0
        snmf = nimfa.Snmf(np.matrix(data),rank=k, beta = betaa ,max_iter=1000,  W = w_r, H = h_r,version='r',eta=1., min_residuals  = 0.0001)
        snmf_fit = snmf()
        W = snmf_fit.fit.W
        H = snmf_fit.fit.H
        spsh[t,b] = snmf_fit.fit.sparseness()[1]
        spsw[t,b] = snmf_fit.fit.sparseness()[0]
        evals[t,b] = snmf_fit.fit.evar()
        kl[t,b] = snmf_fit.distance(metric='kl')
        bet[t,b] = calculate_error(D, W, H)
        rss[t,b] = snmf_fit.fit.rss()
        

        #W_min_mat[t,b] = np.matrix.min(W)
        #W_min_mat[t,0] = k
        #H_min_mat[t,b] = np.matrix.min(H)
        #H_min_mat[t,0] = k
        aw = 'U:/SF_data/01o/W_' +str(k)+'_Beta_'+str(betaa)+'_.csv'
        pd.DataFrame(W).to_csv(aw, sep=',')   
        ah = 'U:/SF_data/01o/H_' +str(k)+'_Beta_'+str(betaa)+'_.csv'
        pd.DataFrame(H).to_csv(ah, sep=',') 
    #ahh = 'G:/uiuc/igl data/ParkingData/SFpark_MeterPaymentData (1)/vkcode vs nimfa/beta conclusion/H_min_.csv'
    #pd.DataFrame(H_min_mat).to_csv(ahh, sep=',') 
    #aww = 'G:/uiuc/igl data/ParkingData/SFpark_MeterPaymentData (1)/vkcode vs nimfa/beta conclusion/W_min_.csv'
    #pd.DataFrame(W_min_mat).to_csv(aww, sep=',')   
        if t == 25 and b == 50:
            pd.DataFrame(bet).to_csv('U:/SF_data/01o/bet_half.csv', sep=',') 
            print("#################################")
            print(betaa)
            

    pd.DataFrame(bet).to_csv('U:/SF_data/01o/bet.csv', sep=',') 
    pd.DataFrame(spsh).to_csv('U:/SF_data/01o/spsh.csv', sep=',') 
    pd.DataFrame(spsw).to_csv('U:/SF_data/01o/spsw.csv', sep=',') 
    pd.DataFrame(rss).to_csv('U:/SF_data/01o/rss.csv', sep=',') 
    pd.DataFrame(evals).to_csv('U:/SF_data/01o/evals.csv', sep=',') 
    pd.DataFrame(kl).to_csv('U:/SF_data/01o/kl.csv', sep=',') 
    print("#################################")
    print(betaa)
    
   