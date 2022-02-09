import numpy as np
from scipy import optimize
import pandas as pd

class costing:
    def data_clean(self,data):
        data1 = data.to_numpy() # Convert to numpy array

        y = data1[:,0]
        X1 = data1[:,1:]
        
        X = (X1 - np.min(X1)) / (np.max(X1) - np.min(X1))
        
        return X, y

    def sigmoid(self,z):
        
        g = 1/(1 + np.exp(-z))
        
        return g
    
    # Regularized cost function definition
    def costFunctionReg(self,w,X,y,lambda_):
        
        m,n = np.shape(X)
        
        if y.dtype == bool:
            y = y.astype(int)
        
        J = 0
        grad = np.zeros(w.shape)
        
        h = self.sigmoid(np.dot(X,w))

        J = -np.dot(y,np.log(h))*(1/m) - np.dot((1-y),np.log(1-h))*(1/m) + (lambda_/(2*m))*(np.sum(w[1:]**2))

        grad = np.transpose(X)@(h - y)/m + np.insert(w[1:],0,0)*lambda_/m

        return J, grad
    
    # Prediction based on trained model
    # Use sigmoid function to calculate probability rounded off to either 0 or 1
    def predictOneVsAll(self,all_w,X,num_labels):
                
        m = np.shape(X)[0]
        p = np.zeros(m)
                
        p = np.argmax(self.sigmoid(X.dot(all_w.T)), axis = 1)
        
        return p
    
    # Optimization defintion
    def minCostFun(self, train_data, iters):
        
        X_train, y_train = self.data_clean(train_data)
        lambda_ = 0.1         # Regularization parameter
        num_labels = 10
        m, n = X_train.shape
        all_w = np.zeros((num_labels, n + 1))
        
        X_train = np.concatenate([np.ones((m, 1)), X_train], axis=1)
        
        for c in np.arange(num_labels):
            initial_w = np.zeros(n + 1)
            options = {'maxiter' : iters}
            res = optimize.minimize(self.costFunctionReg,
                                    initial_w,
                                    (X_train,(y_train==c),lambda_),
                                    jac = True,
                                    method='CG',
                                    options=options)
                                    
            all_w[c] = res.x
            message = res.message
            print(message)
        
        all_w = np.round(all_w,3)       # Optimized weights rounded off to 3 decimal places
        self.all_w = all_w
       
        pred = self.predictOneVsAll(all_w, X_train,num_labels)
        acrcy = np.round((np.mean(pred == y_train) * 100),3)      # Training set accuracy (in %) rounded off to 3 decimal places
        
        return all_w, acrcy
    
    # Calculate testing accuracy
    def TestingAccu(self, test_data):
        
        X_test, y_test = self.data_clean(test_data)
        
        num_labels = 10
        
        m, n = X_test.shape
        X_test = np.concatenate([np.ones((m, 1)), X_test], axis=1)
        
        pred1 = self.predictOneVsAll(self.all_w, X_test,num_labels)
        acrcy_test = np.round((np.mean(pred1 == y_test) * 100),3)
        
        return acrcy_test