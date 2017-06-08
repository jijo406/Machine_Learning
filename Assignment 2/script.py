import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    
    k = np.unique(y)
    d = X.shape[1]
    means=np.zeros((d,k.size))
    covmat = np.zeros(d)
    
    for i in range(0, k.size):
        td = X[np.where(y==i+1)[0]]
        means[:,i] = np.mean(td, 0).T
    #print(means)
    covmat = np.cov(X,rowvar=0)
    #print(covmat)
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    
    col = X.shape[1]
    a = np.unique(y)
    means=np.zeros((col,a.size))
    covmats=[np.zeros((col,col))] * a.size
    for i in range(0,a.size):
        means[:,i]=np.mean(X[np.where(y==i+1)[0]],0).T
        covmats[i]=np.cov(X[np.where(y==i+1)[0]],rowvar=0)

    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    
    d=np.zeros((Xtest.shape[0],means.shape[1]))
    mean = means.T
    for i in range(means.shape[1]):
        diff = Xtest - means[:,i]
        s = np.sum(diff * np.dot(diff,inv(covmat)),1)
        d[:,i] = (np.exp(-1 * s/2))/(sqrt(pi * 2)*(np.power(det(covmat),0.5)))
    
    ypred = np.argmax(d,1) + 1
    acc = 100 * np.mean(ypred == ytest.T)
    
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    
    d = np.zeros((Xtest.shape[0],means.shape[1]))

    for i in range(means.shape[1]):
        diff = Xtest - means[:,i]
        dot = np.dot(diff,inv(covmats[i]))
        s = np.sum(diff * dot,1)
        d[:,i] = (np.exp(-1 * s/2))/(sqrt(pi * 2) * np.power(det(covmats[i]),0.5))

    ypred = np.argmax(d,1)
    ypred = ypred + 1
    acc = 100 * np.mean(ypred == ytest.T)
    
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
    # IMPLEMENT THIS METHOD   
    
    xdot = np.dot(X.T,X)  
    w = np.dot(np.linalg.inv(xdot),np.dot(X.T,y))                                            
    return w    

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD                                                   
    laiden = lambd*np.identity(X.shape[1]) 
    xdot = np.dot(X.T,X)
    w = np.linalg.inv(xdot + laiden)    
    w = np.dot(w, np.dot(X.T,y))

    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    sumation = np.sum(np.square(ytest - np.dot(Xtest,w)))
    n = Xtest.shape[0]
    mse = 1/n * (sumation)
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD                                             
    w = np.mat(w)
    secondpt = (y-(np.dot(X,w.T)))
    firstpt = secondpt.T
    error = ((np.dot(firstpt,secondpt))/2) + (((lambd) * (np.dot(w,w.T)))/2)

    grad1 = np.dot(np.dot(X.T,X),w.T)
    grad2 = np.dot(X.T,y)
    error_grad = ((grad1 - grad2) + lambd*(w.T)).T
    error_grad = np.hstack(np.array(error_grad))
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1)) 
	
    # IMPLEMENT THIS METHOD

    Xd = np.ones((x.shape[0], p + 1))
    for i in range(p+1):
        Xd[:,i] = np.power(x,i)
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

print("\nproblem 1")
# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)
mle_t = testOLERegression(w,X,y)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)
mle_ti = testOLERegression(w_i,X_i,y)

print("\nproblem 2")

print('MSE without intercept for Test '+str(mle))
print('MSE with intercept for Test '+str(mle_i))
print('MSE without intercept for Training '+str(mle_t))
print('MSE with intercept for Training '+str(mle_ti))
#print('Weight without intercept '+str(w))
#print('Weight with intercept '+str(w_i))


# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1


print("\nproblem 3")

mse_t = np.min(mses3_train)
mse = np.min(mses3)
print("MSE for Train Data " + str(mse_t))
print("MSE for Test Data " + str(mse))
#print('Weight for Reidge Regression '+str(w_l))

lambdaopt3 = lambdas[np.argmin(mses3)]

w_l = learnRidgeRegression(X_i,y,lambdaopt3) # weights learnt by ridge regression with optimal lambda

plt.plot(np.linspace(0,64,num = 65),w_i)
plt.plot(np.linspace(0,64,num = 65),w_l)
plt.legend(('OLE regression weights', 'Ridge regression weights \n ( lamdaopt: '+str(lambdaopt3)+')'))

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1

print('\nProblem 4')
fig = plt.figure(figsize=[12,6])
mse_t = np.min(mses4_train)
mse = np.min(mses4)
print("MSE for Train Data " + str(mse_t))
print("MSE for Test Data " + str(mse))

plt.subplot(1, 2, 1)


plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(mses4)]# REPLACE THIS WITH lambda_opt estimated from Problem 3

print("\nproblem 5")
print("Lambda_opt "+ str(lambda_opt))
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

mse_t = np.min(mses5_train)
mse = np.min(mses5)
print("MSE for Train Data " + str(mse_t))
print("MSE for Test Data " + str(mse))

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization\n ( lamdaopt: '+str(lambda_opt)+')'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization\n ( lamdaopt: '+str(lambda_opt)+')'))
plt.show()
