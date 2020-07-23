#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import random

#Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing Thompson Sampling
N = 500
d = 10
ads_selected = []
n_rewards1 = [0] * d
n_rewards0 = [0] * d
total_reward = 0

for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(n_rewards1[i] + 1, n_rewards0[i] + 1)
        if (random_beta > max_random):
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if (reward == 1):
        n_rewards1[ad] += 1
    else:
        n_rewards0[ad] += 1
    total_reward += reward
    
#Visualizing the results (Histogram)
plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
        
    