import numpy as np
from random import choice
import matplotlib.pyplot as plt

def train_perceptron_separable(training_data):
    X = training_data[0]
    y = training_data[1]
    model_size = X.shape[1]
    """
    Use random weight generation to observe weight init effect, otherwise feel free
    to use np.zeros for consistency
    """
    #w = np.zeros(model_size)
    w=np.random.rand(model_size)
    is_converged=False
    iteration = 1
    while not is_converged:
        # compute results according to the hypothesis
        for i in range (len(X)):
            result= np.dot(X[i],w)
            print(result)
            if (np.sign(result)!= y[i]):
                w = w + y[i]*X[i]
                break
            elif (i==len(X)-1):
                is_converged=True
                break
            else:
                continue                 
        # get incorrect predictions (you can get the indices)

        # Check the convergence criteria (if there are no misclassified
        # points, the PLA is converged and we can stop.)

        # Pick one misclassified example.

        # Update the weight vector with perceptron update rule

        iteration += 1
    print("Iteration number is: " + str(iteration))
    return w

def train_perceptron_non_separable(training_data):
    X = training_data[0]
    y = training_data[1]
    model_size = X.shape[1]
    w = np.zeros(model_size)
    is_converged=False
    iteration = 1
    while not is_converged:
        for i in range (len(X)):
            result= np.dot(X[i],w)
            '''
            Convergence threshold trick here: if absolute value of dot product is smaller than a certain value,
            then it has to be closer to the decision boundary, which makes is eligible for being neglected
            '''
            if ((np.sign(result)!= y[i] and abs(result)>0.05) or abs(result)==0):
                w = w + y[i]*X[i]
                break
            elif (i==len(X)-1):
                is_converged=True
                break
            else:
                continue                 
      

        iteration += 1
    print("Iteration number is: " + str(iteration))
    return w

def print_prediction(model,data,labels):
    '''
    Print the predictions given the dataset and the learned model.
    :param model: model vector
    :param data:  data points
    :return: nothing
    '''
    result = np.matmul(data,model)
    predictions = np.sign(result)
    #for i in range(len(data)):
        #print("{}: {} -> {} and real label is: {}".format(data[i][1:], result[i], predictions[i],labels[i]))
    return predictions

def plot_prediction(data,model,predictions):
        plt.scatter([data[i][1] for i in range(len(data)) if predictions[i]==1],[data[i][2] for i in range(len(data)) if predictions[i]==1],marker="o",c="green")
        plt.scatter([data[i][1] for i in range(len(data)) if predictions[i]!=1],[data[i][2] for i in range(len(data)) if predictions[i]!=1],marker="x",c="red")
        x1 = np.linspace(-0.5, 1.2, 50)    
        x2 = -(model[1]*x1 + model[0])/model[2]
        plt.plot(x1,x2)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Vector points on 2d and decision boundary")
        plt.show()

def plot_raw_data(data,labels):
        print(data)
        print(labels)
        plt.scatter([data[i][1] for i in range(len(data)) if labels[i]==1],[data[i][2] for i in range(len(data)) if labels[i]==1],marker="o",c="green")
        plt.scatter([data[i][1] for i in range(len(data)) if labels[i]!=1],[data[i][2] for i in range(len(data)) if labels[i]!=1],marker="x",c="red")
        plt.show()

    
if __name__ == '__main__':

    large_data=np.load("./PLA_data-20240416/data_large.npy")
    large_label=np.load("./PLA_data-20240416/label_large.npy")
    small_data=np.load("./PLA_data-20240416/data_small.npy")
    small_label=np.load("./PLA_data-20240416/label_small.npy")
    rnd_data_large = [np.array(large_data),np.array(large_label)]
    rnd_data_small = [np.array(small_data),np.array(small_label)]

    print("Results for small data set")
    trained_model = train_perceptron_separable(rnd_data_small)
    predictions=print_prediction(trained_model, rnd_data_small[0],rnd_data_small[1])
    plot_prediction(rnd_data_small[0],trained_model,predictions)


    '''
    Non separable part below might be commented to run small data set repetitively 
    for comparing weight initialization impact. Uncomment the code below if you want to see
    the results for the large data set.
    '''
    print("Results for large data set")
    trained_model_non_separable = train_perceptron_non_separable(rnd_data_large)
    predictions_non_separable=print_prediction(trained_model_non_separable, rnd_data_large[0],rnd_data_large[1])
    plot_prediction(rnd_data_large[0],trained_model_non_separable,predictions_non_separable)

    """
    plot_raw_data is defined to show that large dataset is not separable, shown in report
    """
    #plot_raw_data(rnd_data_large[0],rnd_data_large[1])
   

