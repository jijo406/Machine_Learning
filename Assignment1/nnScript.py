import operator
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time 
import matplotlib.pyplot as plt

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return 1.0/(1.0+np.exp(-z))  # your code here

def FeedFoward(w1,w2,data):
    #get bias
    bias = np.ones(data.shape[0])
    #add bias
    data = np.column_stack([data,bias])
    #feed hidden
    zj = sigmoid(np.dot(data,w1.T))
    #zj bias
    bias2 = np.ones(zj.shape[0])
    #add bias
    zj = np.column_stack([zj, bias2])
    #feed output
    ol = sigmoid(np.dot(zj,w2.T))

    return data,zj, ol  

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Your code here.
    # print(len(test_data))
    c = []
    selected_feature = []
    for i in range(len(train_data[1])):
        if np.std(train_data[:,i]) < 0.05 and np.std(validation_data[:,i]) < 0.05:
                c.append(i)
        else: 
            selected_feature.append(i)

    train_data = np.delete(train_data,c, axis = 1)
    validation_data = np.delete(validation_data,c, axis = 1)
    test_data = np.delete(test_data,c, axis = 1)
    

    return selected_feature, train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    #a = []
    #z = []

    #b = []
    
    training_data,zj,ol = FeedFoward(w1,w2,training_data)
       
    #training_label = np.reshape(len(training_data),10) 
    #print(training_label)
    #print(ol[0])
    
    index = np.arange(training_label.shape[0],dtype="int")
    label = np.zeros((training_label.shape[0],n_class))
    label[index,training_label.astype(int)]=1

    training_label = label
    #print(training_label)

    #delta and gradience
    delta = np.subtract(ol,training_label)
    #print(delta)
    #grad_w2
    grad_w2 = np.dot(delta.T,zj)
    
    #grad_w1
    w2_pt1 = (1-zj)*zj
    w2_pt2 = np.dot(delta,w2)
    w2_pt3 = (w2_pt1 * w2_pt2)
    grad_w1= np.dot(w2_pt3.T,training_data)
    #grad_w1 = np.dot(((1-zj)*zj* (np.dot(delta,w2))).T,training_data)

    #take care of hidden row, (Note that we do not compute the gradient for the weights at the bias hidden node.) 
    grad_w1 = np.delete(grad_w1, n_hidden,0)
    
    
    #obj_val
    n = training_data.shape[0]
    oil1 = np.log(ol)
    yil1 = training_label
    firstpt = np.multiply(yil1,oil1)
    oil2 = np.log(np.subtract(1.0,ol))
    yil2 = np.subtract(1.0,training_label)
    secondpt = np.multiply(yil2,oil2)
    ji = np.sum(-1*(firstpt + secondpt))
    obj_val = (1/n) * np.sum(ji)

    #obj_val Regularization
    
    w1_sum = np.sum(np.square(w1)) 
    w2_sum = np.sum(np.square(w2))
    obj_val_reg = (lambdaval/(2*n)) * (w1_sum + w2_sum)
    
    #obj_grad Regularization
    
    grad_w1 = (grad_w1 + (w1*lambdaval))/n
    grad_w2 = (grad_w2 + (w2*lambdaval))/n

    #final Obj_val
    obj_val = obj_val + obj_val_reg
    #print(obj_val)
    
    """
    index = 0
    for i, x in enumerate(train_data):
        index += 1
    print(index)

    # for i,x in enumerate(w1):
    #   print(i,x)

    print(len(w1))
    print(len(w1[1]))
    print(len(train_data))
    print(len(train_data[0]))

    #n_class = 10
    #n_hidden = 50
    #n_input = 784

    w1 row =50
    w1 column =785
    td row = 50000
    td column = 784
    """

    """for x in range(0, n_hidden):
        a.append(np.dot(w1[x][:-1], train_data[x]))
        z.append(sigmoid(a[x]))
    print(a)
    print(z)"""

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = obj_grad

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])

    data, zj, ol = FeedFoward(w1,w2,data)
    
    labels = np.argmax(ol, 1)

    return labels


"""**************Neural Network Script Starts here********************************"""

selected_features, train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network
"""n_input = 5
    n_hidden = 3
    n_class = 2
    training_data = np.array([np.linspace(0,1,num=5),np.linspace(1,0,num=5)])
    training_label = np.array([0,1])
    lambdaval = 0
    params = np.linspace(-5,5, num=26)
    args = (n_input, n_hidden, n_class, training_data, training_label, lambdaval)
    objval,objgrad = nnObjFunction(params, *args)
    print(objval)
    print(objgrad)
    """
start_time = time.time()

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
for i in range(0,60):
    lambdaval = i

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

elapsed_time = start_time - time.time()

print(elapsed_time)

obj = [selected_features, n_hidden, w1, w2, lambdaval]
# selected_features is a list of feature indices that you use after removing unwanted features in feature selection step
import pickle

pickle.dump(obj, open('params.pickle', 'wb'))
