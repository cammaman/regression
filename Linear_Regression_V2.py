import numpy 
import numpy as np
import time

start_time = time.time()

filename = 'data.txt'

#import data as array, ensure you're in the right place (GetWorkingDirectory or similar)
data = np.array(numpy.loadtxt(filename, dtype='float', delimiter=","))

#%%
#define columns of variables required, change for x features > 1
xcol = 0
ycol = 1

#parse variables into separate matrices

xvals = data[:,xcol]
yvals = data[:,ycol]

#should be equal to xvals (m as number of training examples)

m = len(yvals)

#%%

def AddBias(SampleSize, XArray):
    
    X_Bias = np.c_[np.ones(len(XArray)), XArray]
       
    return X_Bias


def gradientDescent(x, y, theta, alpha, m, Iterations):
    
    xTrans = x.transpose()
    for i in range(0, Iterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
       
        
        cost = np.sum(loss ** 2) / (2 * m)
        
        #Keeps you nice and updated
        print("Iteration %d | Cost: %f" % (i, cost))
        
        gradient = np.dot(xTrans, loss) / m
        # update thetas
        theta = theta - alpha * gradient
        
    return theta
    

# Create the x matrix with a column of ones for bias units

Bias_X = AddBias(m, xvals)

#make sure all the parameters have the right shape, m=mL and n =n(x)L
m, n = np.shape(Bias_X)

#Not too many due to the brute inefficiency of gradient descent
Iterations= 1000000

# try a few different alphas
alpha = 0.00005

#One parameter for every feature in input array
theta = np.ones(n)


theta = gradientDescent(Bias_X, yvals, theta, alpha, m, Iterations)

print(theta)
print("--- %s seconds ---" % (time.time() - start_time))
