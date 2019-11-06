#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 10:57:38 2018

@author: hafizimtiaz
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb

# helper functions
# helper functions
def mySigmoid(A):
    return np.divide(1, 1 + np.exp(- A))

def myActivation(A):
    num = np.exp(A) - np.exp(-A)
    den = np.exp(A) + np.exp(-A)
    func = np.divide(num, den)
    return func

def relu(x):    
    return np.maximum(0,x)

def myXentropy(A, Y):
    m = A.shape[1]
    return -(1.0/m) * (np.dot(np.log(A), Y.T) + np.dot(np.log(1-A), (1-Y).T))


def myNormalize(X):
    N = X.shape[1]
    nrms = np.sqrt(np.diag(np.dot(X.T, X)))
    max_nrm = nrms.max()
    mult = (1.0 / max_nrm) * np.diag(np.ones([N,]))
    return np.dot(X, mult)

def myClipping(A, B, clip = 1):
    #computes np.dot(A, B.T) with clipping
    N = A.shape[1]
    res = 0
    for n in range(N):
        a = A[:, n]
        b = B[:, n]

        if (len(a) == 1) or (len(b) == 1):
            tmp = (1.0 / N) * a * b
            nrm = np.linalg.norm(tmp)
        else:
            tmp = (1.0 / N) * np.outer(a, b)
            nrm = np.linalg.norm(tmp, 'fro')

        tmp = tmp * 1.0 / np.max((1, nrm * 1.0 /clip))
        res += tmp
        
    return res

def load_dataset(N = 10000, D = 50):
    """
    This function generates synthetic data and splits into train/test sets.
    
    Returns:
    train_set_x -- train examples, shape = (D, N_tr)
    train_set_y -- train example labels, shape = (1, N_tr)
    test_set_x -- test examples, shape = (D, N_ts)
    test_set_y -- test example labels, shape = (1, N_ts)
    classes -- actual class labels
    """

    N_tr = np.int(N * 70 / 100)
    
    dist = 0.5
    X1 = np.random.randn(D, N // 2) - dist
    labels1 = np.zeros([1, N // 2])
    X2 = np.random.randn(D, N // 2) + dist
    labels2 = np.ones([1, N // 2])
    
    X = np.concatenate((X1, X2), axis = 1)
    labels = np.concatenate((labels1, labels2), axis = 1)
    
    # normalize features for differential privacy
    X = myNormalize(X)
    
    ids = np.arange(0,N)
    np.random.shuffle(ids)
    
    Xall = X[:, ids]
    Yall = labels[0, ids]
    Yall = Yall.reshape([1, N])
    
    # train and test sets
    X_tr = Xall[:, :N_tr]
    Y_tr = Yall[:, :N_tr]
    X_ts = Xall[:, N_tr:]
    Y_ts = Yall[:, N_tr:]
    
    return X_tr, Y_tr, X_ts, Y_ts


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = np.int(m * 1.0 / mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k + 1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
#        end = m - mini_batch_size * np.floor(m * 1.0 / mini_batch_size)
        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size:]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches 

def initialize_parameters(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1 # integer representing the number of layers
     
    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        
    return parameters

def initialize_adam(parameters) :
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    Returns: 
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """
    
    L = np.int(len(parameters) / 2) # number of layers in the neural networks
    v = {}
    s = {}
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

        s["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l + 1)])
        s["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l + 1)])

    
    return v, s  
    
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.999, gamma=1e-8):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    gamma -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    
    L = np.int(len(parameters) / 2)          # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        ### START CODE HERE ### (approx. 2 lines)
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]
        ### END CODE HERE ###

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        ### START CODE HERE ### (approx. 2 lines)
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))
        ### END CODE HERE ###

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        ### START CODE HERE ### (approx. 2 lines)
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.power(grads['dW' + str(l + 1)], 2)
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.power(grads['db' + str(l + 1)], 2)
        ### END CODE HERE ###

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        ### START CODE HERE ### (approx. 2 lines)
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))
        ### END CODE HERE ###

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, gamma". Output: "parameters".
        ### START CODE HERE ### (approx. 2 lines)
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v_corrected["dW" + str(l + 1)] / np.sqrt(s["dW" + str(l + 1)] + gamma)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v_corrected["db" + str(l + 1)] / np.sqrt(s["db" + str(l + 1)] + gamma)
        ### END CODE HERE ###

    return parameters, v, s


def compute_cost(a3, Y):
    
    """
    Implement the cost function
    
    Arguments:
    a3 -- post-activation, output of forward propagation
    Y -- "true" labels vector, same shape as a3
    
    Returns:
    cost - value of the cost function
    """
    m = Y.shape[1]
    
    logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    cost = 1.0 / m * np.sum(logprobs)
    
    return cost

def forward_propagation(X, parameters):
    """
    Implements the forward propagation (and computes the loss) presented in Figure 2.
    
    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape ()
                    b1 -- bias vector of shape ()
                    W2 -- weight matrix of shape ()
                    b2 -- bias vector of shape ()
    
    Returns:
    loss -- the loss function (vanilla logistic loss)
    """
    
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    z1 = np.dot(W1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = mySigmoid(z2)
    
    cache = (z1, a1, W1, b1, z2, a2, W2, b2)
    
    return a2, cache

def backward_propagation(X, Y, cache, epsilon = 0, delta = 0, num_sites = 1, mode = 'cape'):
    """
    Implement the backward propagation presented in figure 2.
    
    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
    cache -- cache output from forward_propagation()
    epsilon, delta -- privacy parameters
    num_sites -- number of sites for distributed computing
    mode -- 'cape' or 'conv'
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    D, m = X.shape
    (z1, a1, W1, b1, z2, a2, W2, b2) = cache
    
    dz2 = 1.0 / m * (a2 - Y)
    dW2 = myClipping(dz2, a1) # np.dot(dz2, a1.T)
    dW2 = np.reshape(dW2, W2.shape)
    db2 = myClipping(dz2, np.ones([1, m])) # np.sum(dz2, axis=1, keepdims = True)
    db2 = np.reshape(db2, b2.shape)
    
    da1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = myClipping(dz1, X) # np.dot(dz1, X.T)
    dW1 = np.reshape(dW1, W1.shape)
    db1 = myClipping(dz1, np.ones([1, m])) # np.sum(dz1, axis=1, keepdims = True)
    db1 = np.reshape(db1, b1.shape)
    
    
    # add noise if dp
    if epsilon != 0:
        D2, D1 = W2.shape

        if mode == 'cape':
            sigma = (1.0 / (m * epsilon)) * np.sqrt(2 * np.log(1.25/delta))
        else:
            sigma = (num_sites * 1.0 / (m * epsilon)) * np.sqrt(2 * np.log(1.25/delta))

        noise_dw2 = sigma * np.random.randn(D2, D1)
        dW2 += noise_dw2
        
        noise_db2 = sigma * np.random.randn(D2)
        db2 += noise_db2
        
        noise_dw1 = sigma * np.random.randn(D1, D)
        dW1 += noise_dw1
        
        noise_db1 = sigma * np.random.randn(D1, 1)
        db1 += noise_db1
    
    
    gradients = {"dz2": dz2, "dW2": dW2, "db2": db2,
                 "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}
    
    return gradients

def predict(X, y, parameters):
    """
    This function is used to predict the results of a  n-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    p = np.zeros((1,m), dtype = np.int)
    
    # Forward propagation
    a2, caches = forward_propagation(X, parameters)
    
    # convert probas to 0/1 predictions
    p = 1.0 * np.greater(a2, 0.5)
    acc = np.mean((p[0,:] == y[0,:]))
    
    return p, acc

def model(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9,
          beta1=0.9, beta2=0.999, gamma=1e-8, num_epochs=10000, print_cost=True,
          epsilon = 0, delta = 0, num_sites = 1, mode = 'cape'):
    """
    2-layer neural network model
    
    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    layers_dims -- python list, containing the size of each layer
    learning_rate -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates 
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
    gamma -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs
    epsilon, delta -- privacy parameters
    num_sites -- number of sites for distributed computing
    mode -- 'cape' or 'conv'
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """

    costs = []                       # to keep track of the cost
    t = 0                            # initializing the counter required for Adam update
    seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours
    
    # Initialize parameters
    parameters = initialize_parameters(layers_dims)

    # Initialize the optimizer
    v, s = initialize_adam(parameters)        
    
    # Optimization loop
    for i in range(num_epochs):
        
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            a2, caches = forward_propagation(minibatch_X, parameters)

            # Compute cost
            cost = compute_cost(a2, minibatch_Y)

            # Backward propagation
            grads = backward_propagation(minibatch_X, minibatch_Y, caches, epsilon, delta, num_sites, mode)

            # Update parameters
            t = t + 1 # Adam counter
            parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2,  gamma)
                
        # Print the cost every 1000 epoch
        if print_cost and i % 100 == 0:
            print("Cost after epoch %i: %f" % (i, cost))
            costs.append(cost)
#        if print_cost and i % 100 == 0:
#            costs.append(cost)
                
#    # plot the cost
#    plt.plot(costs)
#    plt.ylabel('cost')
#    plt.xlabel('epochs (per 100)')
#    plt.title("Learning rate = " + str(learning_rate))
#    plt.show()

    return parameters, costs
##################


#%% optimization
alpha = 0.01
#N_all = [1000, 2000, 3000, 4000, 5000, 7000, 8000, 10000]
N_all = [1000, 2000, 3000, 4000, 5000, 7000, 8000, 10000]
epsilon = 5e-1
delta = 0.01
num_sites = 4
nensemble = 10
maxIter = 500

acc_nonp_tr = np.zeros([nensemble, len(N_all)])
acc_nonp_ts = np.zeros([nensemble, len(N_all)])
acc_conv_tr = np.zeros([nensemble, len(N_all)])
acc_conv_ts = np.zeros([nensemble, len(N_all)])
acc_cape_tr = np.zeros([nensemble, len(N_all)])
acc_cape_ts = np.zeros([nensemble, len(N_all)])
costs_nonp = np.zeros([nensemble, np.int(maxIter/100)])
costs_conv = np.zeros([nensemble, np.int(maxIter/100)])
costs_cape = np.zeros([nensemble, np.int(maxIter/100)])

for n_id in range(len(N_all)):
    N = N_all[n_id]
    train_X, train_Y, test_X, test_Y = load_dataset(N, 50)
    layers_dims = [train_X.shape[0], 5, 1]
    batch_size = np.min((np.int(train_X.shape[1]/10), 128))
    for rep in range(nensemble):
#        pdb.set_trace()
        # non-priv computation
        parameters, costs_nonp_tmp = model(train_X, train_Y, layers_dims, optimizer="adam", learning_rate=0.01, mini_batch_size=batch_size, num_epochs=maxIter)
    
        # Predict
        Y_pred_tr, acc_tr = predict(train_X, train_Y, parameters)
        Y_pred_ts, acc_ts = predict(test_X, test_Y, parameters)
        
        print("Train percent acc: %0.4f" % (acc_tr * 100))    
        print("Test percent acc: %0.4f" % (acc_ts * 100))
        
        acc_nonp_tr[rep, n_id] = acc_tr * 100
        acc_nonp_ts[rep, n_id] = acc_ts * 100
        costs_nonp[rep, :] = costs_nonp_tmp
        
        # dp computation            
        # cape        
        parameters, costs_cape_tmp = model(train_X, train_Y, layers_dims, optimizer="adam", learning_rate=0.01, mini_batch_size=batch_size, num_epochs=maxIter, \
                           epsilon = epsilon, delta = delta, num_sites = num_sites, mode = 'cape')
    
        # Predict
        Y_pred_tr, acc_tr = predict(train_X, train_Y, parameters)
        Y_pred_ts, acc_ts = predict(test_X, test_Y, parameters)
        
        print("CAPE Train percent acc: %0.4f" % (acc_tr * 100))    
        print("CAPE Test percent acc: %0.4f" % (acc_ts * 100))
    
        acc_cape_tr[rep, n_id] = acc_tr * 100
        acc_cape_ts[rep, n_id] = acc_ts * 100
        costs_cape[rep, :] = costs_cape_tmp
        
        # conv
        parameters, costs_conv_tmp = model(train_X, train_Y, layers_dims, optimizer="adam", learning_rate=0.01, mini_batch_size=batch_size, num_epochs=maxIter, \
                           epsilon = epsilon, delta = delta, num_sites = num_sites, mode = 'conv')
    
        # Predict
        Y_pred_tr, acc_tr = predict(train_X, train_Y, parameters)
        Y_pred_ts, acc_ts = predict(test_X, test_Y, parameters)
        
        print("CONV Train percent acc: %0.4f" % (acc_tr * 100))    
        print("CONV Test percent acc: %0.4f" % (acc_ts * 100))
    
        acc_conv_tr[rep, n_id] = acc_tr * 100
        acc_conv_ts[rep, n_id] = acc_ts * 100
        costs_conv[rep, :] = costs_conv_tmp
        

#%% save results
res_filename = 'synth_vs_samples_D50_adam_v2'
np.savez(res_filename, acc_nonp_tr, acc_nonp_ts, acc_conv_tr, acc_conv_ts, acc_cape_tr, acc_cape_ts, N_all, epsilon, delta, costs_nonp, costs_conv, costs_cape)
    