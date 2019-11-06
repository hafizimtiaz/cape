#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 08:34:22 2018

@author: hafizimtiaz
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb

#%% vs epsilon
plt.figure(figsize=(5,2))
# plot results of synth
res_filename = 'synth_cost_decay_D50'
arr = np.load(res_filename + '.npz')
cost_nonp = arr['arr_8'].mean(axis = 0)
cost_conv = arr['arr_9'].mean(axis = 0)
cost_cape = arr['arr_10'].mean(axis = 0)


itr = np.arange(1, 201)
#plt.subplot(1,3,1)
plt.semilogx(itr, cost_nonp, itr, cost_conv, itr, cost_cape)
plt.xlabel('Iteration (in 100)')
plt.ylabel('Average cost')
plt.title('Synthetic $(N = 10k, \epsilon = 0.01)$')
plt.legend(['non-priv', 'conv', 'cape'], loc = 'best')


#plt.savefig("cost_decay_nn.pdf", format = "pdf")
plt.show()

