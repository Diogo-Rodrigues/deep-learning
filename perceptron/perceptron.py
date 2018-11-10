import numpy as np
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(1000)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    print('X: ' + str(X))
    print('W: ' + str(W))
    print('b: ' + str(b))
    return stepFunction((np.matmul(X,W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    # Fill in code
    for i in range(len(X)):
        y_hat = prediction(X[i], W, b)
        if y[i] - y_hat == 1:
            W[0] = W[0] + learn_rate*X[i][0]
            W[1] = W[1] + learn_rate*X[i][1]
            b = b + learn_rate
        elif y[i] - y_hat == -1:
            W[0] = W[0] - learn_rate*X[i][0]
            W[1] = W[1] - learn_rate*X[i][1]
            b = b - learn_rate
    return W, b
    
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
        print(boundary_lines)
    return boundary_lines

f = "data.csv"
X = []
y = []

with open(f) as csv:
  for line in csv:
    x1, x2, label = line.split(',') # if you want to act imediately
    X.append([float(x1), float(x2)])
    y.append(float(label.rstrip()))

X = np.array(X)
y = np.array(y)

boundary_lines = trainPerceptronAlgorithm(X, y)
print(boundary_lines)

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

x1 = X.T[0]
x2 = X.T[1]

col = np.where(y,'r','b')

plt.scatter(x1, x2, c=col, s=5, linewidth=0)

for line_data in boundary_lines:
    # Plot a single line
    xs = [i for i in range(len(line_data))]
    ys = line_data
    plt.plot(xs, ys, color='green')

plt.show()