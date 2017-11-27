#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 20:26:10 2017

@author: steve
"""

# GET PACKAGES
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import random

# CREATE DATA
x = np.linspace(0,100,num = 101)
y = np.sin(x/10) + (x/70)**2


# plot curve
plt.plot(x,y)

# Do svm to compare
from sklearn import svm
xTrain = np.linspace(0,100,num = 101)
yTrain = np.sin(xTrain/10) + (xTrain/70)**2

# make y training data noisy
randomness = .2
for i in range(len(yTrain)-1):
    yTrain[i] = yTrain[i] + (randomness*random.randint(-3,3))

# plot original data and noisy training data
plt.scatter(xTrain,yTrain,edgecolors = 'black',cmap = plt.cm.coolwarm)

# Fit svm to noisy data
svmModel = svm.SVR(kernel = 'rbf',C=1e3,gamma = .001)
xTrain = xTrain.reshape(101,1)
yTrain = yTrain.reshape(101,1)
svmModel.fit(xTrain,yTrain)

# use svm to predict function
x = x.reshape(101,1)
yPredict = svmModel.predict(x)

# plot original and predicted
plt.figure()
original = plt.plot(x,y,label = 'Original')
predicted = plt.plot(x,yPredict,label = 'SVM Predicted')
plt.legend()
plt.show()

# BFunction to build and train the model on training data
#def buildNN(numPasses,hiddenLayerDimensions,xy):

# INITIATE MODEL AND PARAMETERS
hiddenLayerDimension = 5

# Set number of passes of gradient descent to run
numPasses =20000

# reshape y from n, to n,1
y = y.reshape(len(y),1)
yTrain = yTrain.reshape(len(yTrain),1)
    
# Gradient descent parameters 
epsilon = 0.001 # learning rate for gradient descent
regLambda = 0.01 # regularization strength
 
num_examples = len(xTrain) # training set size
inputDimension = xTrain.shape[1] # input layer dimensionality
outputDimension = xTrain.shape[1] # output layer dimensionality
    

# Take random values for weights, biases. Use randn normal distributino w/ mean, variance^2
   # np.random.seed(0)
W1 = np.random.randn(inputDimension,hiddenLayerDimension)/np.sqrt(inputDimension)
W2 = np.random.randn(hiddenLayerDimension,outputDimension)/np.sqrt(hiddenLayerDimension)
b1 = np.zeros((1, hiddenLayerDimension))
b2 = np.zeros((1, outputDimension))

# Initiate model
model = {}

# Gradient Descent (batch gradient descent)
print('Training...')
for i in range(numPasses):
    
  #  print(i,' ',W1)
    # Forward Pass, take parameters (weights, biases) and move forward through graph computing node values
    z1 = x.dot(W1) + b1 # .dot does matrix multiplication
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = z2 
    
    # partial derivative of Loss w/ respect to prediction yhat
   # partialL_partialyHat = 2*(a2-yTrain)/num_examples
    partialL_partialyHat = (a2-y)
    if partialL_partialyHat[0] == partialL_partialyHat[0]:
        #print(partialL_partialyHat)
        df = a2-y
        print(df[5])
        
    
    # weight gradients
    dW2 = a1.T.dot(partialL_partialyHat) # compute partial L/partial W2
    dW1 = x.T.dot(partialL_partialyHat.dot(W2.T)*(1-np.power(a1,2))) # compute partialL/partial W1
    
    # Bias gradients, biases only have 1 column, they are just one value, so take sum 
   # db2 = np.sum(partialL_partialyHat,axis = 0,keepdims = True) # sum probbtilities matrix along row
    db2 =  partialL_partialyHat
   # db1 = np.sum(partialL_partialyHat.dot(W2.T)*(1-np.power(a1,2)),axis = 0,keepdims = True) # sum along row
    db1 = partialL_partialyHat.dot(W2.T)*(1-np.power(a1,2))
    
    # Regularization
    dW2 = dW2 + regLambda*W2
    dW1 = dW1 + regLambda*W1
    
    # Update weights and biases
    W2 = W2 + (-epsilon*dW2)
    W1 = W1 + (-epsilon*dW1)
    b2 = b2 + (-epsilon*db2)
    b1 = b1 + (-epsilon*db1)
    
    # Assign new parameters to the model
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

# Get weights and biases from model
W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']

# Do a forward pass for prediction
z1 = xTrain.dot(W1) + b1 # .dot does matrix multiplication
a1 = np.tanh(z1)
z2 = a1.dot(W2) + b2
a2 = z2 
prediction= a2

plt.figure()
plt.plot(xTrain,prediction)
plt.title('Neural Net Prediction')
#original = plt.plot(x,y,label = 'Original')
plt.legend()
    