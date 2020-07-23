#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Importing the libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implemeting UCB
N = 10000
d = 10
ads_selected = []
n_selections = [0] * d 
sum_rewards = [0] * d
total_reward = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (n_selections[i] > 0):
            avg_reward = sum_rewards[i] / n_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / n_selections[i])
            upper_bound = avg_reward + delta_i
        else:
           upper_bound = 1e400
        if (upper_bound > max_upper_bound):
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    n_selections[ad] += 1
    reward = dataset.values[n, ad]
    sum_rewards[ad] += reward
    total_reward += reward

#Visualizing the results
plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
