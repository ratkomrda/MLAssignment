# Ported from Andrew Ngs matlab examples from his machine learning course
# https://www.coursera.org/learn/machine-learning/programming/8f3qT/linear-regression
# Url: https://www.coursera.org/learn/machine-learning/resources/O756o
# Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions
#  in this exericse:
#
#     sigmoid
#     costFunction
#     predict
#     costFunctionReg
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from matplotlib import markers


def sigmoid(z):
    # SIGMOID Compute sigmoid function
    # g = SIGMOID(z) computes the sigmoid of z.
    g = 1 / (1 + np.exp(-z))
    return g


def costFunction(theta, X, y):
    # COSTFUNCTION Compute cost and gradient for logistic regression
    # J = COSTFUNCTION(theta, X, y)
    # computes the cost of using theta as the
    # parameter for logistic regression and the gradient of the cost
    # w.r.t.to the parameters.
    m_local = len(y)
    h = sigmoid(X @ theta)
    diff_sum = np.sum(np.multiply(-y, np.log(h)) - np.multiply((1 - y), np.log(1 - h)))
    J = np.multiply(diff_sum, 1 / m_local)
    return J


def gradient(theta, X, y):
    m_local = len(y)
    h = sigmoid(X @ theta)
    diff = (h - y)
    return X.T @ diff * (1/m_local)


def accuracy(X, y, theta, cutoff):
    pred = [sigmoid(np.dot(X, theta)) >= cutoff]
    acc = np.mean(pred == y)
    print(acc * 100)

# Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.
data = pd.read_csv('..\data\ex2data1.txt', header = None)
X = data.iloc[:,:-1]
y = data.iloc[:,2]


## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the
#  the problem we are working with.
# Instructions: Plot the positive and negative examples on a
#              2D plot, using the option 'k+' for the positive
#               examples and 'ko' for the negative examples.
print(['Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n']);
positive = y == 1
negative = y == 0

Xpositive = plt.scatter(X[positive][0].values, X[positive][1].values, c='red', marker='+')
Xnegative = plt.scatter(X[negative][0].values, X[negative][1].values, c='black', marker='o')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend((Xpositive, Xnegative), ('Admitted', 'Not admitted'))

## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You need to complete the code in
#  costFunction

# Setup the data matrix appropriately, and add ones for the intercept term
(m, n) = X.shape
X = np.hstack((np.ones((m,1)), X))
y = y[:, np.newaxis]

# Initialize fitting parameters
theta = np.zeros((n+1,1))
cost = costFunction(theta, X, y)
grad = gradient(theta, X, y)
print('Cost at initial theta (zeros): \n', cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): \n', grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')


# Compute and display cost and gradient with non-zero theta
test_theta = np.array([[-24], [0.2], [0.2]])
cost_test = costFunction(test_theta, X, y)
grad_test = gradient(test_theta, X, y)

print('Cost at test theta: \n', cost_test)
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta: \n', grad_test)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

## ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.
#  Run fminunc to obtain the optimal theta
# This function returns 3 elements the first contains the solution in this case the optimized theta, the second
# is the number of function evaluations the third is an error code

result = opt.fmin_tnc(func=costFunction, x0=theta, fprime=gradient, args=(X, y.flatten()))
rc = result[2]
if rc != 0:
    exit(rc)
thetaOpt = result[0]

print(thetaOpt)

costOpt = costFunction(thetaOpt[:,np.newaxis], X, y)
# Print theta to screen
print('Cost at theta found by fminunc: \n', costOpt)
print('Expected cost (approx): 0.203\n')
print('theta: \n', thetaOpt)
print('Expected theta (approx):\n');
print(' -25.161\n 0.206\n 0.201\n');


plot_x = [np.min(X[:,1]-2), np.max(X[:,2]+2)]
plot_y = -1/thetaOpt[2]*(thetaOpt[0]
          + np.dot(thetaOpt[1],plot_x))
decision_boun = plt.plot(plot_x, plot_y)

plt.show()

accuracy(X, y.flatten(), thetaOpt, 0.5)