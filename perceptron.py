import numpy as np

def sign(x):
    if x>0:
        return 1
    else:
        return 0

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    n,d=train_set.shape       
    W=np.zeros(d)      #initiate the weight parameters
    b=0

    for j in range(max_iter):
        for i in range(n):
        #idx=np.remainder(i,n)
            if (sign(np.dot(W,train_set[i])+b)!=train_labels[i])&(train_labels[i]==1):
                W+=learning_rate*train_set[i]
                b+=learning_rate
            elif (sign(np.dot(W,train_set[i])+b)!=train_labels[i])&(train_labels[i]==0):
                W-=learning_rate*train_set[i]
                b-=learning_rate

    # return the trained weight and bias parameters
    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set
    L=[]
    
    W,b=trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    for i in range(len(dev_set)):
        L.append(sign(np.dot(W,dev_set[i])+b)) 
    return L

