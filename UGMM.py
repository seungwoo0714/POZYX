#!/usr/bin/env python
# coding: utf-8

# In[84]:


import numpy as np

def h(x,H):
    d = x-H
    d = np.multiply(d,d)
    d = np.sqrt(d.sum(axis=1))
    d_col_1 = d[1]-d[0]
    d_col_2 = d[2]-d[0] 
    d_col_3 = d[3]-d[0]
    di = np.array([d_col_1,d_col_2,d_col_3])
    di = di.T
    return di

class UGMM(object):
    '''Univariate GMM with CAVI'''
    def __init__(self, X, K=2, sigma=1):
        self.X = X
        self.K = K
        self.N = self.X.shape[0]
        self.sigma2 = sigma**2

    def _init(self):
        self.phi = np.random.dirichlet([np.random.random()*np.random.randint(1, 10)]*self.K, self.N)
        print(self.phi)
        K = X[0]
        K = np.append(K,X[SAMPLE])
        K = np.append(K,X[2*SAMPLE])
        K = np.append(K,X[3*SAMPLE])
        print(K)
        self.m = K
        self.s2 = np.ones(self.K) * np.random.random(self.K)
        print('Init mean')
        print(self.m)
        print('Init s2')
        print(self.s2)

    def get_elbo(self):
        t1 = np.log(self.s2) - self.m/self.sigma2
        t1 = t1.sum()
        t2 = -0.5*np.add.outer(self.X**2, self.s2+self.m**2)
        t2 += np.outer(self.X, self.m)
        t2 -= np.log(self.phi)
        t2 *= self.phi
        t2 = t2.sum()
        return t1 + t2

    def fit(self, max_iter=100, tol=1e-10):
        self._init()
        self.elbo_values = [self.get_elbo()]
        self.m_history = [self.m]
        self.s2_history = [self.s2]
        for iter_ in range(1, max_iter+1):
            self._cavi()
            self.m_history.append(self.m)
            self.s2_history.append(self.s2)
            self.elbo_values.append(self.get_elbo())
            if iter_ % 5 == 0:
                print(iter_, self.m_history[iter_])
            if np.abs(self.elbo_values[-2] - self.elbo_values[-1]) <= tol:
                print('ELBO converged with ll %.3f at iteration %d'%(self.elbo_values[-1],
                                                                     iter_))
                break

        if iter_ == max_iter:
            print('ELBO ended with ll %.3f'%(self.elbo_values[-1]))

    def _cavi(self):
        self._update_phi()
        self._update_mu()

    def _update_phi(self):
        t1 = np.outer(self.X, self.m)
        t2 = -(0.5*self.m**2 + 0.5*self.s2)
        exponent = t1 + t2[np.newaxis, :]
        self.phi = np.exp(exponent)
        self.phi = self.phi / self.phi.sum(1)[:, np.newaxis]

    def _update_mu(self):
        self.m = (self.phi*self.X[:, np.newaxis]).sum(0) * (1/self.sigma2 + self.phi.sum(0))**(-1)
        assert self.m.size == self.K
        #print(self.m)
        self.s2 = (1/self.sigma2 + self.phi.sum(0))**(-1)
        assert self.s2.size == self.K

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

import matplotlib.pyplot as plt
import seaborn as sns

bias = 1
std = 2

x = np.array([7.54,4.8])
H = np.array([[7.54,0],[7.54,7.21],[14.14,0],[14.14,7.58]])
H2 = np.matrix([[7.54,0],[7.54,7.21],[14.14,0],[14.14,7.58]])
d = x-H
d = np.multiply(d,d)
d = np.sqrt(d.sum(axis=1))
print(d)

d[0] = d[0] + np.random.normal(bias,std,1)
d[1] = d[1] + np.random.normal(bias,std,1)
d[2] = d[2] + np.random.normal(bias,std,1)
d[3] = d[3] + np.random.normal(bias,std,1)
zd = np.matrix([d[0],d[1],d[2],d[3]])
print(zd)
x_hat = tri(zd,H2)
x_hat = x_hat.T+H[0]
print(x_hat)

SAMPLE = 100
X1 = np.random.normal(loc=d[0]+bias, scale=std, size=SAMPLE)
X2 = np.random.normal(loc=d[1]+bias, scale=std, size=SAMPLE)
X3 = np.random.normal(loc=d[2]+bias, scale=std, size=SAMPLE)
X4 = np.random.normal(loc=d[3]+bias, scale=std, size=SAMPLE)
X1 = X1.T
X2 = X2.T
X3 = X3.T
X4 = X4.T
X = np.append(X1,X2)
X = np.append(X,X3)
X = np.append(X,X4)
print(X)

#for i, mu in enumerate(mu_arr[1:]):
#    X = np.append(X, np.random.normal(loc=mu, scale=1, size=SAMPLE))

fig, ax = plt.subplots(figsize=(15, 4))
sns.distplot(X[:SAMPLE], ax=ax, rug=True)
sns.distplot(X[SAMPLE:SAMPLE*2], ax=ax, rug=True)
sns.distplot(X[SAMPLE*2:SAMPLE*3], ax=ax, rug=True)
sns.distplot(X[SAMPLE*3:SAMPLE*4], ax=ax, rug=True)
import time
start = time.time()
ugmm = UGMM(X, 4)
ugmm.fit()
ugmm.phi.argmax(1)
zd = np.matrix([ugmm.m[0],ugmm.m[1],ugmm.m[2],ugmm.m[3]])
x_hat = tri(zd,H2)
x_hat = x_hat.T+H[0]
print(x_hat)
_check_usage_of_cpu_and_memory()
print("time :", time.time() - start)
#sorted(mu_arr)
#sorted(ugmm.m)
#for i in range(0,3)
x_init = np.array([1,1])

fig, ax = plt.subplots(figsize=(15, 4))
plt.grid()
sns.distplot(X[:SAMPLE], ax=ax, hist=True, norm_hist=True)
sns.distplot(np.random.normal(ugmm.m[0], 1, SAMPLE), color='k', hist=False, kde=True)
sns.distplot(X[SAMPLE:SAMPLE*2], ax=ax, hist=True, norm_hist=True)
sns.distplot(np.random.normal(ugmm.m[1], 1, SAMPLE), color='k', hist=False, kde=True)
sns.distplot(X[SAMPLE*2:SAMPLE*3], ax=ax, hist=True, norm_hist=True)
sns.distplot(np.random.normal(ugmm.m[2], 1, SAMPLE), color='k', hist=False, kde=True)
sns.distplot(X[SAMPLE*3:], ax=ax, hist=True, norm_hist=True)
sns.distplot(np.random.normal(ugmm.m[3], 1, SAMPLE), color='k', hist=False, kde=True)
print(ugmm.m_history)


# In[14]:


print(X[0])


# In[18]:


K = X[0]
K = np.append(K,X[SAMPLE])
K = np.append(K,X[2*SAMPLE])
K = np.append(K,X[3*SAMPLE])
print(K)


# In[ ]:




