#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 14:08:34 2018

@author: hafizimtiaz
"""
import numpy as np

def load_data(m = 1000, n = 10):
    beta = np.random.randn(n, 1)
    X = np.random.randn(n, m)
    b = 2
    y = np.dot(beta.T, X) + b
    y = 1 * (y >= 0.5)
    
    Xts = np.random.randn(n, m // 3)
    yts = np.dot(beta.T, Xts) + b
    yts = 1 * (yts >= 0.5)
    return X, y, beta, b, Xts, yts

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    gp = g * (1-g)
    return g, gp

def relu(z):
    bg = z >= 0
    g = np.zeros_like(z)
    g[bg] = z[bg]
    
    gp = np.zeros_like(z)
    gp[bg] = 1
    return g, gp

def predict(W, b, X):
    L = len(W.keys())
    A = dict()
    Z = dict()
    A[0] = X
    for l in range(1, L):
        Z[l] = np.dot(W[l], A[l-1]) + b[l]
        A[l], _ = relu(Z[l])
        
    Z[L] = np.dot(W[L], A[L-1]) + b[L]
    A[L], _ = sigmoid(Z[L])
    
    return A, Z

def grad(W, b, X, Y):
    L = len(W.keys())
    m = X.shape[1]
    
    A, Z = predict(W, b, X)
    dZ = dict()
    dW = dict()
    db = dict()
    
    dZ[L] = A[L] - Y
    dW[L] = (1/m) * np.dot(dZ[L], A[L-1].T)
    db[L] = (1/m) * np.sum(dZ[L], axis = 1, keepdims = True)
    for l in range(L-1, 0, -1):
        _, gp = relu(Z[l])
        dZ[l] = np.dot(W[l+1].T, A[l+1]) * gp
        dW[l] = (1/m) * np.dot(dZ[l], A[l-1].T)
        db[l] = (1/m) * np.sum(dZ[l], axis = 1, keepdims = True)
    
    return dW, db

def mycosts(W, b, X, Y):
    m = X.shape[1]
    L = len(W.keys())
    A, Z = predict(W, b, X)
    Y_hat = A[L]
    tmp = Y * np.log(Y_hat) + (1-Y) * np.log(1-Y_hat)
    J = (-1/m) * tmp.sum()
    return J

def myerror(W, b, X, Y):
    m = X.shape[1]
    L = len(W.keys())
    A, Z = predict(W, b, X)
    Y_hat = np.round(A[L])
    tmp = (Y_hat != Y).sum()
    return tmp / m


def mymbatch(X, Y, batch_size = 64):
    m = X.shape[1]
    ids = np.arange(m)
    np.random.shuffle(ids)
    X = X[:, ids]
    Y = Y[:, ids]
    
    batches = list()
    st = 0
    en = batch_size
    while en <= m:
        Xb = X[:, st:en]
        Yb = Y[:, st:en]
        batches.append((Xb, Yb))
        st += batch_size
        en += batch_size
    return batches

def update(W, b, dW, db, alpha):
    L = len(W.keys())
    for l in range(1, L+1):
        W[l] -= alpha * dW[l]
        b[l] -= alpha * db[l]
    return W, b

def main(maxItr = 100, alpha = 0.01):
    X_tr, Y_tr, beta0, b0, X_ts, Y_ts = load_data()

    n = X_tr.shape[0]
    layers = [n, 5, 1]
    L = len(layers) - 1
    
    W = dict()
    b = dict()
    for l in range(1, L+1):
        nl = layers[l]
        nlp = layers[l-1]
        W[l] = np.random.randn(nl, nlp)
        b[l] = np.zeros([nl, 1])
    
    costs_tr = list()
    costs_ts = list()
    error = list()
    for itr in range(maxItr):
        batches = mymbatch(X_tr, Y_tr)
        
        for (Xb, Yb) in batches:
            dW, db = grad(W, b, Xb, Yb)
            W, b = update(W, b, dW, db, alpha)
        
        costs_tr.append(mycosts(W, b, X_tr, Y_tr))
        costs_ts.append(mycosts(W, b, X_ts, Y_ts))
        error.append(myerror(W, b, X_ts, Y_ts))
    
    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.plot(costs_tr, 'r--')
    plt.plot(costs_ts, 'b--')
    plt.title('Costs: %0.4f' % costs_tr[-1])
    
    plt.subplot(122)
    plt.plot(error)
    plt.title('Errors: %0.4f' % error[-1])
    plt.tight_layout()
    plt.show()
    return costs_tr, costs_ts, error

if __name__ == '__main__':
#    import pdb
#    pdb.set_trace()
    costs_tr, costs_ts, error = main(maxItr = 50, alpha = 0.05)
    
            