# -*- coding: utf-8 -*-
"""
Created on Fri Sep 08 02:34:09 2017

@author: Vipul Satone
"""
# code for occupancy matrix
# please change the mont and year in line number 37 and 39 and file adress as well
# I will comment this code shortly
import numpy as np
import pandas as pd
from datetime import *
import numpy as np
import math
import time
chunk = pd.read_csv("U:/SF_data/jan_2013.csv")
a = list(chunk.STREET_BLOCK.unique())   
t = 0
fg =[]
b = ['date']
c = ['time']
e = ['month']
w = ['Year']
fg = []
fg1 = w + e + b + c
df = pd.DataFrame(columns = fg1 + a)
time1=[]
for i in range(24):
    for j in range(4):
        p = i*100 + j*15
        a1 = p
        time1.append(a1)
day = range(1,31)
day = np.repeat(day,96)
#@@@@@@@@@@@@@@@@@@@@@@
# please mont numbere here
mon = [1] 
#Please change year here
df['Year'] = 2013
#@@@@@@@@@@@@@@@@@@@@@
mon = np.repeat(mon,96*30)
df['date'] = day 
df['month'] = mon 
df['time']= (list(time1) * 30)
chunk = chunk.drop(['NET_PAID_AMT','PAYMENT_TYPE','PM_DISTRICT_NAME','POST_ID'], axis = 1)
for i in range((len(chunk))):
    print i
    s_t = datetime.strptime(chunk.iloc[i,2], '%d-%b-%Y %H:%M:%S')
    s_dday = int(datetime.strftime(s_t,'%d'))
    s_mmon = int(datetime.strftime(s_t,'%m'))
    s_hour = int(datetime.strftime(s_t,'%H'))
    s_minu = int(datetime.strftime(s_t,'%M'))
    s_t_c = int(s_hour)*100 + int(s_minu)
    e_t = datetime.strptime(chunk.iloc[i,3], '%d-%b-%Y %H:%M:%S')
    e_hour = int(datetime.strftime(e_t,'%H'))
    e_minu = int(datetime.strftime(e_t,'%M'))
    e_t_c = int(e_hour)*100 + int(e_minu)
    name = chunk['STREET_BLOCK'][i]
    lst_day = map(lambda x:x, s_dday == df['date'])
    lst_stime = map(lambda x:x, s_t_c-15 < df['time'])
    lst_etime = map(lambda x:x, e_t_c >= df['time'])
    val = list(np.array(lst_day) * np.array(lst_stime) * np.array(lst_etime))
    nan_ind = [index for index, value in enumerate(val) if value == True]
    df[name][nan_ind] = 1 + df[name][nan_ind].fillna(0)
    not_nan_ind = [index for index, value in enumerate(val) if value == False]     
df.to_csv("U:/SF_data/jan_occ_2013.csv", sep=',')


