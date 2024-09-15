#import libraries
import numpy as np
import pandas as pd
import matplotlib as mp

#read data
path = '1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

# -------------------------------------------------------------------------------

# show data details

# to show first 10 data
print('data.head = \n', data.head(10))
print('='*50)

# to show data description (count, mean, std, min, 25%, 50%, 75%, max)
print('data.describe = \n', data.describe())
print('='*50)

# to plot the data 
data.plot(kind = 'scatter', x = 'Population', y = 'Profit', figsize = (5,5))

# -------------------------------------------------------------------------------

# to insert X0 = 1 to features (for theta 0 or Bias )

data.insert(0,'X0',1)
print('data.head = \n', data.head(10))
print('='*50)

# -------------------------------------------------------------------------------

# to separate features from target

# number of training data (rows) --> m , number of features (columns) --> n
m,n = data.shape

# to remove the last column , the target
n = n - 1

# all rows without the last column
X = data.iloc[:,:n]

# all rows with just the last column
Y = data.iloc[:,n:n+1]

print('Features = \n', X.head(10))
print('='*50)

print('Target = \n', Y.head(10))
print('='*50)

# -------------------------------------------------------------------------------

# to convert from data frame to matrix

X = np.matrix(X.values)
Y = np.matrix(Y.values)

# to get theta or weights or parameters we have n weights theta 0 or bias and other weight for each feature
weights = np.matrix(np.zeros(n))

print('X \n',X)
print('X.shape = ' , X.shape)
print('='*50)
print('weights \n',weights)
print('weights.shape = ' , weights.shape)
print('='*50)
print('Y \n',Y)
print('Y.shape = ' , Y.shape)
print('='*50)

# -------------------------------------------------------------------------------

# to compute cost function

def Cost_Function(X,Y,weights):
    m,n = X.shape
    z = np.power(((X * weights.T) - Y), 2)
    cost = np.sum(z) / (2 * m)
    return cost

print('Cost Function(X,Y,weights,m) = ' , Cost_Function(X,Y,weights))
print('='*50)

# -------------------------------------------------------------------------------

# to compute Gradient Descent function

def Gradient_Descent_Function(X,Y,weights,alpha,iteration):
    # temp matrix that will compute the new weights
    temp = np.matrix(np.zeros(weights.shape))
    
    # to get all costs at each iteration
    cost = np.zeros(iteration)
    
    m,n = X.shape
    
    for i in range(iteration):
        
        error = (X * weights.T) - Y
        
        for j in range(n):
            # to get the term inside summation in the gradient descent rule
            term = np.multiply(error,X[:,j])
            
            # to calculate gradient descent
            temp[:,j] = weights[:,j] - alpha * np.sum(term) / m
        
        # to update weights and costs 
        weights = temp
        cost[i] = Cost_Function(X, Y, weights)
        
    return cost , weights

# learning rate
alpha = 0.01

# iterations
iteration = 1000

cost , weights = Gradient_Descent_Function(X,Y,weights,alpha,iteration)

print('Costs after each iterations:\n ' ,cost )
print('='*50)
print('Parameters after Gradient Descent:\n', weights )
print('='*50)

# -------------------------------------------------------------------------------

# get best fit line

# to get a x values from minimum and maximum population value
x = np.linspace(data.Population.min(), data.Population.max(), 100)

# to calculate best fit 
BestFitLine = weights[0,0] + weights[0,1] * x

# draw the line

fig, ax = mp.pyplot.subplots(figsize=(5,5))
ax.plot(x, BestFitLine, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')


# draw error graph

fig, ax = mp.pyplot.subplots(figsize=(5,5))
ax.plot(np.arange(iteration), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

# -------------------------------------------------------------------------------

# to predict new value

def Linear_Regression_Predection(Feature, weights):

    return weights[0,0] + weights[0,1] * Feature

print('predection value of 25 = ',Linear_Regression_Predection(25, weights))