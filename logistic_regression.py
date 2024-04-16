import numpy as np
from random import choice
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo 
  



def map_targets(Y):
     for index,row in Y.iterrows():
          if(row["Class"]=="Osmancik"):
               row["Class"]=1
          else:
               row["Class"]=-1
          
class Train_Options:
  def __init__(self, learning_rate, threshold):
    self.learning_rate = learning_rate
    self.threshold = threshold            

def train_gd(X,Y,is_regularized,options=Train_Options(0.1,0.21),lambda_param=0,max_iter=200):
    num_rows,num_columns= X.shape
    W=np.zeros(num_columns)
    learning_rate = options.learning_rate
    threshold=options.threshold
    is_converged=False
    iteration = 1
    
    while not is_converged:
        loss=0
        weight_update=0
        # compute error and update component for each sample
        for index,row in X.iterrows():
             y=Y.iloc[row.name]["Class"]
             loss=(loss*index+ np.log(1+np.exp(-y*np.dot(row,W))))/(index+1)
             weight_update=(weight_update*index + y*row*learning_rate/(1+np.exp(y*np.dot(row,W))))/(index+1)
        if(is_regularized):
            loss=loss + lambda_param*(np.linalg.norm(W)**2)
            weight_update=weight_update + 2*lambda_param*np.linalg.norm(W)
        if(loss<threshold):
             print(loss)
             is_converged=True
        if(iteration==max_iter):
             print(loss)
             is_converged=True
        
        else:
             W=W+weight_update    
        iteration += 1
    print("iteration no: "+str(iteration))
    return W

def sigmoid(z): 
    return 1 / (1 + np.exp(-z))

def calculate_accuracy(W,X,Y):
     correct_count=0
     incorrect_count=0
     for index,row in X.iterrows():
          z=np.dot(W,row)
          prediction= sigmoid(z)
          if(prediction>=0.5):
               if(Y.iloc[row.name]["Class"]==1):
                correct_count+=1
               else:
                incorrect_count+=1
          if(prediction<0.5):
               if(Y.iloc[row.name]["Class"]==-1):
                correct_count+=1
               else:
                incorrect_count+=1
     print("Correct guesses:" + str(correct_count))
     print("Incorrect guesses"+ str(incorrect_count))
     print("Total samples: " + str(X.shape[0]))
     print("Success Rate: " + str((correct_count/(incorrect_count+correct_count))*100))

def five_fold(X,Y,lambda_array):
     np.random.seed(1)
     indices = np.arange(len(X))
     np.random.shuffle(indices)

     # Split the shuffled indices into 5 folds
     num_folds = 5
     fold_size = len(X) // num_folds
     fold_indices = [indices[i*fold_size:(i+1)*fold_size] for i in range(num_folds)]

     for lambda_param in lambda_array: 
      print("Starting cross validation for regularization param: "+str(lambda_param))
     # Perform 5-fold cross-validation
      for fold in range(num_folds):
          # Get indices for training and testing sets
          test_indices = fold_indices[fold]
          train_indices = np.concatenate([fold_indices[i] for i in range(num_folds) if i != fold])
    
          # Split the data into training and testing sets
          train_data, test_data = X.iloc[train_indices], X.iloc[test_indices]
          print("Cross Validation Result for lambda" + str(lambda_param) +" and validation step "+str(fold))
          W = train_gd(train_data,Y,True,lambda_param=lambda_param,max_iter=1000,options=Train_Options(0.01,0.21))
          print("Results for Train Data")
          calculate_accuracy(W,train_data,Y)
          print("Results for Test Data")
          calculate_accuracy(W,test_data,Y)
          
    

    
    
if __name__ == '__main__':

   # fetch dataset 
    rice_cammeo_and_osmancik = fetch_ucirepo(id=545) 
  
    # data (as pandas dataframes) 
    X = rice_cammeo_and_osmancik.data.features 
    Y = rice_cammeo_and_osmancik.data.targets 

    
    # metadata 
    #print(rice_cammeo_and_osmancik.metadata) 
  
    # variable information 
    #print(rice_cammeo_and_osmancik.variables)   

    #Normalize features
    mean = X.mean()
    std_dev = X.std()
    standardized_X = (X - mean) / std_dev

    np.random.seed(1)
    indices = np.arange(len(standardized_X))
    np.random.shuffle(indices)

     # Split the data for test and train samples for regular GD
    num_folds = 5
    fold_size = len(standardized_X) // num_folds
    fold_indices = [indices[i*fold_size:(i+1)*fold_size] for i in range(num_folds)]
    test_indices = fold_indices[1]
    train_indices = np.concatenate([fold_indices[i] for i in range(num_folds) if i != 1])
    train_data, test_data = standardized_X.iloc[train_indices], standardized_X.iloc[test_indices]
    map_targets(Y)
 
    print("Results for Non-Regularized GD")
    W=train_gd(train_data,Y,False,Train_Options(0.01,0.21))
    print("Results for Train Data")
    calculate_accuracy(W,train_data,Y)
    print("Results for Test Data")
    calculate_accuracy(W,test_data,Y)
 
    five_fold(standardized_X,Y,[-5,-2,-1,-0.1,-0.02])
   

