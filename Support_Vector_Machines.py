#SVMs are used for small data sets with relatively fewer outliers
#It is a much faster way of classification (and sometimes regression) than nueral nets

import numpy as np
from matplotlib import pyplot as plt


#[X value, Y value, Bias term]
X=np.array([[-2,4,-1],
			[4,1,-1],
			[1,6,-1],
			[2,4,-1],
			[6,2,-1]])

#associated output labels
Y=np.array([-1, -1, 1, 1, 1])

#using the hinge loss function for classification 


def svm_sgd_plot(X, Y):
    #Initialize our SVMs weight vector with zeros (3 values)
    w = np.zeros(len(X[0]))
    #The learning rate
    eta = 1
    #how many iterations to train for
    epochs = 100000

    #training part, gradient descent part
    for epoch in range(1,epochs):
        error = 0
        for i, x in enumerate(X):
            #misclassification
            if (Y[i]*np.dot(X[i], w)) < 1:
                #misclassified update for ours weights
                w = w + eta * ( (X[i] * Y[i]) + (-2  *(1/epoch)* w) )
                error = 1
            else:
                #correct classification, update our weights
                w = w + eta * (-2  *(1/epoch)* w)
    for d, sample in enumerate(X):
      # Plot the negative samples
      if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
      # Plot the positive samples
      else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

     # Add our test samples
    plt.scatter(2,2, s=120, marker='_', linewidths=2, color='yellow')
    plt.scatter(4,3, s=120, marker='+', linewidths=2, color='blue')

    # Print the hyperplane calculated by svm_sgd()
    x2=[w[0],w[1],-w[1],w[0]]
    x3=[w[0],w[1],w[1],-w[0]]

    x2x3 =np.array([x2,x3])
    X,Y,U,V = zip(*x2x3)
    ax = plt.gca()
    ax.quiver(X,Y,U,V,scale=1, color='blue')
    plt.show()

svm_sgd_plot(X,Y)