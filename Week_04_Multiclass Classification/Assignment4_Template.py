import numpy as np
from scipy import optimize
import pandas as pd

class costing:
    def data_clean(self,data): # 'data' is a pandas dataframe
        
        
        X =                     # Normalized 'X' (numpy array)
        y =                     # numpy array
        
        return X, y

    def sigmoid(self,z):
        
        
        
        return g
    
    # Regularized cost function definition
    def costFunctionReg(self,w,X,y,lambda_):
        
        
        
        
        J =             # Cost 'J' should be a scalar
        grad =          # Gradient 'grad' should be a vector
        
        return J, grad
    
    # Prediction based on trained model
    # Use sigmoid function to calculate probability rounded off to either 0 or 1
    def predictOneVsAll(self,all_w,X,num_labels):
                
        
        
        p =     # 'p' should be a vector of size equal to that of vector 'y'
        
        return p
    
    # Optimization defintion
    def minCostFun(self, train_data): #'train_data' is a pandas dataframe
        
        lambda_ = 0.1         # Regularization parameter
        iters = 4000
        w_ini =               # Intialize 'w' for all classes as zero
        
        
        
        all_w =        # Optimized weights (size = 10 X 785) rounded off to 3 decimal places
               
        acrcy =        # Training set accuracy (in %) rounded off to 3 decimal places (Ans ~ 93.2)
        
        return all_w, acrcy
    
    # Calculate testing accuracy
    def TestingAccu(self, test_data): #'test_data' is a pandas dataframe
        
        
        
        acrcy_test =    # Training set accuracy (in %) rounded off to 3 decimal places (Ans ~ 86.667)
        
        return acrcy_test