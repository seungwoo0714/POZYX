#!/usr/bin/env python
# coding: utf-8

# In[119]:


import numpy as np
from scipy import stats
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

class GMM(object):
    def __init__(self, X, k=1):
        # dimension
        X = np.asarray(X)
        self.m, self.n = X.shape
        self.data = X.copy()
        # number of mixtures
        self.k = k
        
    def _init(self):
        # init mixture means/sigmas
        self.mean_arr = np.asmatrix(np.random.random((self.k, self.n)))
        self.sigma_arr = np.array([np.asmatrix(np.identity(self.n)) for i in range(self.k)])
        self.phi = np.ones(self.k)/self.k
        self.w = np.asmatrix(np.empty((self.m, self.k), dtype=float))
        #print(self.mean_arr)
        #print(self.sigma_arr)
    
    def fit(self, tol=1e-4):
        self._init()
        num_iters = 0
        ll = 1
        previous_ll = 0
        while(ll-previous_ll > tol):
            previous_ll = self.loglikelihood()
            self._fit()
            num_iters += 1
            ll = self.loglikelihood()
            print('Iteration %d: log-likelihood is %.6f'%(num_iters, ll))
        print('Terminate at %d-th iteration:log-likelihood is %.6f'%(num_iters, ll))
    
    def loglikelihood(self):
        ll = 0
        for i in range(self.m):
            tmp = 0
            for j in range(self.k):
                #print(self.sigma_arr[j])
                tmp += sp.stats.multivariate_normal.pdf(self.data[i, :], 
                                                        self.mean_arr[j, :].A1, 
                                                        self.sigma_arr[j, :]) *\
                       self.phi[j]
            ll += np.log(tmp) 
        return ll
    
    def _fit(self):
        self.e_step()
        self.m_step()
        
    def e_step(self):
        # calculate w_j^{(i)}
        for i in range(self.m):
            den = 0
            for j in range(self.k):
                num = sp.stats.multivariate_normal.pdf(self.data[i, :], 
                                                       self.mean_arr[j].A1, 
                                                       self.sigma_arr[j]) *\
                      self.phi[j]
                den += num
                self.w[i, j] = num
            self.w[i, :] /= den
            assert self.w[i, :].sum() - 1 < 1e-4
            
    def m_step(self):
        for j in range(self.k):
            const = self.w[:, j].sum()
            self.phi[j] = 1/self.m * const
            _mu_j = np.zeros(self.n)
            _sigma_j = np.zeros((self.n, self.n))
            for i in range(self.m):
                _mu_j += (self.data[i, :] * self.w[i, j])
                _sigma_j += self.w[i, j] * ((self.data[i, :] - self.mean_arr[j, :]).T * (self.data[i, :] - self.mean_arr[j, :]))
                #print((self.data[i, :] - self.mean_arr[j, :]).T * (self.data[i, :] - self.mean_arr[j, :]))
            self.mean_arr[j] = _mu_j / const
            self.sigma_arr[j] = _sigma_j / const
            print(self.mean_arr)
            
def tri(zd, H):
    zd = zd.T
    H=H-H[0]
    tmp = zd[0]
    tmp = tmp * tmp
    zd = np.delete(zd,(0), axis=0)
    r1=np.multiply(zd,zd)
    Ha=np.delete(H,(0), axis=0)
    K=np.multiply(Ha,Ha)
    K=K.sum(axis=1)
    b= 0.5*(K-r1+tmp) 
    S_inv = np.linalg.inv(Ha.T*Ha)*Ha.T
    x_hat = S_inv*b
    return x_hat

import os
import psutil

def _check_usage_of_cpu_and_memory():
    
    pid = os.getpid()
    py  = psutil.Process(pid)
    
    cpu_usage   = os.popen("ps aux | grep " + str(pid) + " | grep -v grep | awk '{print $3}'").read()
    cpu_usage   = cpu_usage.replace("\n","")
    
    memory_usage  = round(py.memory_info()[0] /2.**30, 2)
    
    print("cpu usage\t\t:", cpu_usage, "%")
    print("memory usage\t\t:", memory_usage, "%")

bias = 0
std = 1

x = np.array([7.54,4.8])
H = np.array([[7.54,0],[7.54,7.21],[14.14,0],[14.14,7.58]])
H2 = np.matrix([[7.54,0],[7.54,7.21],[14.14,0],[14.14,7.58]])
d = x-H
d = np.multiply(d,d)
d = np.sqrt(d.sum(axis=1))

d_col_1 = d[0]
d_col_2 = d[1] 
d_col_3 = d[2]
d_col_4 = d[3]
real_arr = np.array([d_col_1,d_col_2,d_col_3,d_col_4])
print(real_arr)
zd = np.matrix([d_col_1,d_col_2,d_col_3,d_col_4])
x_hat = tri(zd,H2)
x_hat = x_hat.T+H[0]
print(x_hat)

SAMPLE = 1000
zd_col_tmp_1 = d[0] + np.random.normal(bias,std,1)
zd_col_tmp_2 = d[1] + np.random.normal(bias,std,1)
zd_col_tmp_3 = d[2] + np.random.normal(bias,std,1)
zd_col_tmp_4 = d[3] + np.random.normal(bias,std,1)
X1 = zd_col_tmp_1 
X2 = zd_col_tmp_2
X3 = zd_col_tmp_3
X4 = zd_col_tmp_4
X = np.array([zd_col_tmp_1,zd_col_tmp_2,zd_col_tmp_3,zd_col_tmp_4])
print(X)

zd_col_1 = d[0] + np.random.normal(bias,std,SAMPLE-1)
zd_col_2 = d[1] + np.random.normal(bias,std,SAMPLE-1)
zd_col_3 = d[2] + np.random.normal(bias,std,SAMPLE-1)
zd_col_4 = d[3] + np.random.normal(bias,std,SAMPLE-1)
mu_arr = np.array([zd_col_1,zd_col_2,zd_col_3,zd_col_4])
X = np.append(X,mu_arr,axis = 1)
X1 = np.append(X1,zd_col_1)
X2 = np.append(X2,zd_col_2)
X3 = np.append(X3,zd_col_3)
X4 = np.append(X4,zd_col_4)
X_rev = np.transpose(X)
print(X_rev)
        
#X = np.random.multivariate_normal([0, 3], [[0.5, 0], [0, 0.8]], 20)
#X = np.vstack((X, np.random.multivariate_normal([20, 10], np.identity(2), 50)))
#X.shape
t = range(0,SAMPLE)
fig, ax = plt.subplots(figsize=(15, 4))
sns.distplot(X1, ax=ax, rug=True)
sns.distplot(X2, ax=ax, rug=True)
sns.distplot(X3, ax=ax, rug=True)
sns.distplot(X4, ax=ax, rug=True)
import time
start = time.time()
gmm = GMM(X_rev)
gmm.fit()
dist_1 = gmm.mean_arr[0,0]
dist_2 = gmm.mean_arr[0,1]
dist_3 = gmm.mean_arr[0,2]
dist_4 = gmm.mean_arr[0,3]
zd = np.matrix([dist_1,dist_2,dist_3,dist_4])
x_hat = tri(zd,H2)
x_hat = x_hat.T+H[0]
print(x_hat)
_check_usage_of_cpu_and_memory()
print("time :", time.time() - start)
print(gmm.mean_arr)
fig, ax = plt.subplots(figsize=(15, 4))
plt.grid()
sns.distplot(np.random.normal(gmm.mean_arr[[0],[0]], 1, SAMPLE), color='k', hist=False, kde=True)
sns.distplot(np.random.normal(gmm.mean_arr[[0],[1]], 1, SAMPLE), color='k', hist=False, kde=True)
sns.distplot(np.random.normal(gmm.mean_arr[[0],[2]], 1, SAMPLE), color='k', hist=False, kde=True)
sns.distplot(np.random.normal(gmm.mean_arr[[0],[3]], 1, SAMPLE), color='k', hist=False, kde=True)
sns.distplot(X1, ax=ax, rug=True)
sns.distplot(X2, ax=ax, rug=True)
sns.distplot(X3, ax=ax, rug=True)
sns.distplot(X4, ax=ax, rug=True)


# In[43]:


X_rev[range(0,SAMPLE)][0]


# In[105]:


print(gmm.mean_arr[0,0])


# In[112]:


print(zd[0,0])


# In[ ]:




