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


# Function to plot decision boundary for sklearn model
def plotBoundary(xy,model,title):
    # create mesh grid to plot in, size [xmin-xmax],[ymin-ymax]
    x_min, x_max = xy[:, 0].min() - 1, xy[:, 0].max() + 1 # min/max of x ax 
    y_min, y_max = xy[:, 1].min() - 1, xy[:, 1].max() + 1 # min/max of y ax 
    meshStepSize = .02 # step size
    xx, yy = np.meshgrid(np.arange(x_min, x_max, meshStepSize), np.arange(y_min, y_max, meshStepSize)) # make mesh grid
    
    # Assign a prediction to each point in the mesh corresponding to classes of data points
    meshPredict = model.predict(np.c_[xx.ravel(), yy.ravel()]) # np.c concatenates along column axis
    meshPredict = meshPredict.reshape(xx.shape)
    
    # Put predictions into color plot
    plt.figure()
    plt.contourf(xx, yy, meshPredict, cmap=plt.cm.coolwarm)
    plt.axis('tight')
    
    # Also plot training points
    plt.scatter(xy[:,0], xy[:,1],c = label,cmap = plt.cm.coolwarm, edgecolor = 'black')
    plt.title(title)

# --------------------------------------SOME NOTES ON NN-------------------------------------------
# Nerual net is just adaptive basis function
# Nonlinear function from set of inputs --> set of outputs, controlled by vector w of adjustable parameters
# Want to transform inputs into a new space but dont want to manually set up basis function to do so
# Adaptive basis will continuously change basis functions to improve, say phi is family of basis functions
# Train and do SGD on phi, make nonlinear data linear then we can do regression.

# With feed forward networks, there weight space symmetries, multiple weight vectos that
# lead to the same mapping
# Num of inputs/outputs determined by dimensionality of dataset

# If M controls num of parameters (weights and biases), there is an optimal M that gives
# the best generalization performance ie the best point between under/overfitting.
# One way to find M could be to plot sum of squares error vs num of hidden units in the
# network, then choose solution with smallest validation set error.
# Orrr could a large value for M then use a regularization term to the error
# function in order to control complexity 
# Simple reglarizer is a quadratic one, pick lambda, weight decay

# Instead of regularization could do early stopping.
# DUring training, it is doing an iterative reduction of the error function
# So take a validation set and measure error (during training), itll decrease at first,
# but then start to increase once it starts to overfit. 
# Want to stop training at point of of smallest error (w/ respect to validation set)

# To calculate gradient of loss function, use back propogation.
# Update weights w/ SGD


# For this code:
# Iput will be coordinates, output is what class it is.
# nonlinear activation functions can be relu, tanh, sigmoid
# activation of output is softmax
# Softmax: p_j = exp(x_j) / sum(exp(x_k))
# where p is class probability (output of the unit j) and x_j and x_k represent total inputs to units j and k
# Can use cross entropy loss for loss function. If incorrect class predicted, more loss
# Use gradient descent to minimize this loss function. Try to learn SGD, also decay learning rate over time
# Gradient Descent takes gradients of loss function w/ respect to parameters (weights, biases)
# These gradients are done using back propogation (make computational graph, automatic differentiation)
#
#--------------------------------------------------------------------------------------------------------

# THIS CODE:

# -----------------------LOSS FUNCTION---------------------------------
# Loss function will be cross entropy loss: L(y,yhat) = -1/N sum_n sum_i ( y_n,i*log(yhat_n,i) )
# Sums over training, bigger loss for misclassification
# cross entropy common choice of loss function with softmax outputs

#---------------------- NERUAL NET SETUP--------------------------------
# With one hidden layer we have weights/bias from input to hidden layer W1,b1
# and weights/bias from hidden layer to output W2/b2 then
# at hidden layer we multiply inputs by weights, add bias, and put into activation function
# z1 = xW1 + b1, a1 = activationFunc(z1)
# a is output of layer, which becomes inputs for next layer:
# z2 - a1W2 + b2, a2 = softmax(z2)
# This is for only one hidden layer so output prediction is yhat = a2

#----------------------GRADIENT DESCENT-----------------------------------
# Adaptive basis, we are tryint to learn weights/biases
# using batch gradient descent, try to learn sgd
# Inputs will be gradients, derivatives of Loss function wrt W1,b1,W2,b2, so need to take 4 partials
# Use backpropogation algo which finds gradients starting with output
# See notes for derivation of graidents:
# partial L/partial W2 = (yhat-y)*a1
# partial L/partial B2 = (yhat-y)
# partial L/partial W1 = W2(yhat-y)*(1-tanh(z1)^2)*x
# partial L/partial b1 = W2(yhat-y)*(1-tanh(z1)^2)
# REMEMBER: W1 represents vector of weights from input to hidden layer
#           W2 reoresents vector of weights from hidden layer to output
#           Number of nodes in hidden layer dictates size of Ws
#           Size will be W1: 2xn,   W2: nx2 since W1 taken 2 inputs into n hidden layers
#           and W2 takes n number of hidden layers to 2 outputs (prob of inputs)


# Finally lets actually do it

# BFunction to build and train the model on training data
def buildNN(numPasses,hiddenLayerDimensions,xy):
    
    # INITIATE MODEL AND PARAMETERS
        
    # Gradient descent parameters 
    epsilon = 0.01 # learning rate for gradient descent
    regLambda = 0.01 # regularization strength
     
    num_examples = len(xy) # training set size
    inputDimension = xy.shape[1] # input layer dimensionality
    outputDimension = xy.shape[1] # output layer dimensionality
        
    
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
        z1 = xy.dot(W1) + b1 # .dot does matrix multiplication
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        expScores = np.exp(z2)
        a2 = expScores / np.sum(expScores, axis = 1, keepdims = True) 
        # so a2 = softmax(z2)
        # So softmax takes exponent of all z2 values, in this case there are 2 columns
        # since there are 2 classes. Then divides each of the 2 values in each row by the 
        # sum of the exponents in the row, basically giving a percentage that each value in the row makes up 
        # based on the sum of exponents of the row.
        # Output is same size, each row is a datapoint, has two columns, probabilities of each class
        # View notes to see softmax written out and explained
        
        # Probabilities of each class
        probabilities = a2
        
        
        # Backpropogation, use the differentiated equations and forward pass values to compute gradients
    
    #####    delta3 = probabilities
    #####    delta3[range(num_examples), label] -= 1 # delta 3 has [probability1 probability2] for each row, corresponding
        # to prob of class 0 and prob of class 1. This line looks at the correct column in the row. Label 
        # has to actual label which is 0 or 1, so if the correct value is 0 we look at column 0, if the 
        # correct value for that row is 1 we look at column 1. We then take the value in thalt column and 
        # subtract by 1, which will at most give zero and at smallest give -1. We want gradient or partial
        # derivative of L / partial weight to be zero since that would mean we hit a minima. If the probability
        # for the column value is 1, then we do 1-1 = 0 meaning the partial derivative is zero which is ideal.
        # This makes sense since for each row we want the correct class to have a probability of 1, meaning we 
        # are absolutely sure that it is the correct class.
        
        # Size of W2 is numHiddenNodesx2. Rows is num of nodes and columns are num of weightscoming out of that 
        # node. Since we go from hidden layer to y output which is 2, it's numhiddennodesx2. For every training
        # point we only update the weight going to the y of the correct label. The other weight we leave the same
        # So for a single data point that is class 0, our softmax outputs a prediction for class 0 and a 
        # prediction for class 1. We only look at the class 0 prediction, update the dW2 for that value,
        # then update W2 with dW2. 
        
        # The update of dW2 is done like described above, subtracting 1 from the probability so that if
        # the probablity of correct label is 1, the gradient for it becomes zero.
        rowIndex = np.linspace(0,num_examples-1,num=num_examples) # 0:1:num_examples, row numbers
        rowIndex = rowIndex.astype(int)
       #    probabilities[rowIndex,label] = probabilities[rowIndex,label] - 1
        # Another way of doing it could be
            
        # Get Weight gradients
       # probabilities = a2 # [prob0 prob1]
        correctColumn = label # correct label is 0 or 1
        probabilities[rowIndex,correctColumn] = -(1-probabilities[rowIndex,correctColumn]) # get probability value corresponding to correct label, do -(1-that)
        dW2 = a1.T.dot(probabilities) # compute partial L/partial W2
        dW1 = xy.T.dot(probabilities.dot(W2.T)*(1-np.power(a1,2)))  # compute partialL/partial W1
        
        # Bias gradients, biases only have 1 column, they are just one value, so take sum 
        db2 = np.sum(probabilities,axis = 0,keepdims = True) # sum probbtilities matrix along row
        db1 = np.sum((probabilities.dot(W2.T)*(1-np.power(a1,2))),axis = 0,keepdims = True) # sum along row
        
        # Regularization
        dW2 = dW2 + regLambda*W2
        dW1 = dW1 + regLambda*W1
        
        # Update weights amd biases
        W2 = W2 + (-epsilon*dW2)
        W1 = W1 + (-epsilon*dW1)
        b2 = b2 + (-epsilon*db2)
        b1 = b1 + (-epsilon*db1)
        
        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model
        
        
# Function to use the trained nn to predict new data points
def predictNN(model,xy):
    # Prediction is just a forward pass for a single input
    
    # Get weights and biases from model
    W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
    
    # Do Forward Pass
    z1 = xy.dot(W1) + b1 # .dot does matrix multiplication
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    expScores = np.exp(z2)
    a2 = expScores / np.sum(expScores, axis = 1, keepdims = True) 
    predictions = a2 # [prob0 prob1]
    prediction = np.argmax(predictions,axis = 1) # get indicies with max value for each row
    
    return prediction
     
    
       
    
# Function to plot decision boundary for nn
def plotNNBoundary(xy,model):
    # create mesh grid to plot in, size [xmin-xmax],[ymin-ymax]
    x_min, x_max = xy[:, 0].min() - 1, xy[:, 0].max() + 1 # min/max of x ax 
    y_min, y_max = xy[:, 1].min() - 1, xy[:, 1].max() + 1 # min/max of y ax 
    meshStepSize = .02 # step size
    xx, yy = np.meshgrid(np.arange(x_min, x_max, meshStepSize), np.arange(y_min, y_max, meshStepSize)) # make mesh grid
    
    # Assign a prediction to each point in the mesh corresponding to classes of data points
    meshPredict = predictNN(model,np.c_[xx.ravel(), yy.ravel()]) # np.c concatenates along column axis
    meshPredict = meshPredict.reshape(xx.shape)
    
    # Put predictions into color plot
    plt.figure()
    plt.contourf(xx, yy, meshPredict, cmap=plt.cm.coolwarm)
    plt.axis('tight')
    
    # Also plot training points
    plt.scatter(xy[:,0], xy[:,1],c = label,cmap = plt.cm.coolwarm, edgecolor = 'black')
    plt.title('Neural Net Decision Boundary')


# CREATE DATA
# make data not linearly separable
xy, label = sklearn.datasets.make_moons(100, noise=0.2) # 100 samples, make std dev of gaussian noise almost zero
plt.scatter(xy[:,0], xy[:,1],c = label,cmap = plt.cm.coolwarm)
plt.title('Data Points')


# SHOW HOW LOGISTIC REGRESSION /LINEAR MODEL SUCKS ON IT
import sklearn.linear_model as lm
linearModel = lm.SGDClassifier()
linearModel.fit(xy,label)

# Plot decision boudnary for linear model
plotBoundary(xy,linearModel,'Linear Model Decision Boundary') # wow that sucks, lets do nn


# TRY AN SVM w/ KERNEL
from sklearn.svm import SVC
svmRBF = SVC(C = 50,gamma = .4)
svmRBF.fit(xy,label)

# Plot decision boundary for svm
plotBoundary(xy,svmRBF,'SVM Decision Boundary')


# NERUAL NETWORK
# Set number of nodes in hidden layer
hiddenLayerDimension = 50

# Set number of passes of gradient descent to run
numPasses = 20000

model = buildNN(numPasses,hiddenLayerDimension,xy) # train model
plotNNBoundary(xy,model)# plot it


