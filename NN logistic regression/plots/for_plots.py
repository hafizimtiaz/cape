#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 08:34:22 2018

@author: hafizimtiaz
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy.io import savemat

#%% vs epsilon
plt.figure(figsize=(10,4))
plt.tight_layout()
# plot results of synth
res_filename = 'synth_vs_eps_D50_adam_v2'
arr = np.load(res_filename + '.npz')

epsilon_all = arr['arr_6']
acc_nonp_tr = arr['arr_0'].mean(axis = 0) * np.ones([len(epsilon_all), ])
acc_nonp_ts = arr['arr_1'].mean(axis = 0) * np.ones([len(epsilon_all), ])
acc_conv_tr = arr['arr_2'].mean(axis = 0)
acc_conv_ts = arr['arr_3'].mean(axis = 0)
acc_cape_tr = arr['arr_4'].mean(axis = 0)
acc_cape_ts = arr['arr_5'].mean(axis = 0)

tmp = dict()
tmp['acc_nonp_tr'] = acc_nonp_tr
tmp['acc_nonp_ts'] = acc_nonp_ts
tmp['acc_conv_tr'] = acc_conv_tr
tmp['acc_conv_ts'] = acc_conv_ts
tmp['acc_cape_tr'] = acc_cape_tr
tmp['acc_cape_ts'] = acc_cape_ts
savemat(res_filename, tmp)

plt.subplot(1,4,1)
plt.semilogx(epsilon_all, acc_nonp_tr, epsilon_all, acc_conv_tr, epsilon_all, acc_cape_tr)
#plt.xlabel('Privacy parameter $\epsilon$')
plt.ylabel('Train accuracy %')
plt.title('Synthetic $(N = 10k)$')
plt.legend(['non-priv', 'conv', 'cape'], loc = 'upper left')

plt.subplot(1,4,2)
plt.semilogx(epsilon_all, acc_nonp_ts, epsilon_all, acc_conv_ts, epsilon_all, acc_cape_ts)
plt.xlabel('Privacy parameter $\epsilon$')
plt.ylabel('Test accuracy %')


#%% vs samples
# plot results of synth
res_filename = 'synth_vs_samples_D50_adam_v2'
arr = np.load(res_filename + '.npz')

N_all = arr['arr_6']
acc_nonp_tr = arr['arr_0'].mean(axis = 0)
acc_nonp_ts = arr['arr_1'].mean(axis = 0)
acc_conv_tr = arr['arr_2'].mean(axis = 0)
acc_conv_ts = arr['arr_3'].mean(axis = 0)
acc_cape_tr = arr['arr_4'].mean(axis = 0)
acc_cape_ts = arr['arr_5'].mean(axis = 0)
tmp = dict()
tmp['acc_nonp_tr'] = acc_nonp_tr
tmp['acc_nonp_ts'] = acc_nonp_ts
tmp['acc_conv_tr'] = acc_conv_tr
tmp['acc_conv_ts'] = acc_conv_ts
tmp['acc_cape_tr'] = acc_cape_tr
tmp['acc_cape_ts'] = acc_cape_ts
savemat(res_filename, tmp)

plt.subplot(1,4,3)
plt.semilogx(N_all, acc_nonp_tr, N_all, acc_conv_tr, N_all, acc_cape_tr)
#plt.xlabel('Sample size $N$')
plt.ylabel('Train accuracy %')
plt.title('Synthetic $(\epsilon = 0.01)$')

plt.subplot(1,4,4)
plt.semilogx(N_all, acc_nonp_ts, N_all, acc_conv_ts, N_all, acc_cape_ts)
plt.xlabel('Sample size $N$')
plt.ylabel('Test accuracy %')


#plt.savefig("vs_all_nn.pdf", format = "pdf")
plt.show()

