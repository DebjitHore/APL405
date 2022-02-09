# This module builds NN model, train model and plot the results

import torch
import numpy as np
import matplotlib.pyplot as plt

class FunctionApproximator:
    
    # Initializing the class with a constructor
    # self is not a keyword
    def __init__(self, input_dimension, hidden_dimension, output_dimension):
        self.model = self.buildModel(input_dimension, hidden_dimension, output_dimension)
        self.train_cost_history = None
        self.val_cost_history = None
    
    # This method builds NN model (Basically defines the architecture of NN)
    def buildModel(self, input_dimension, hidden_dimension, output_dimension):
        nonlinearity = torch.nn.Sigmoid()  # activation function
        modules = []                    # Initialize modules
        
        # Append modules as first layer between input and hidden layer is created.
        modules.append(torch.nn.Linear(input_dimension, hidden_dimension[0])) 
        # We can directly define the actiavtion function here instead of saving it in line 18.
        modules.append(nonlinearity)
        # This line executes when there is more than 1 hidden layer.
        # It repeats the same functionalities as created between input layer and hidden layer in earlier step.
        for i in range(len(hidden_dimension)-1):
            modules.append(torch.nn.Linear(hidden_dimension[i], hidden_dimension[i+1]))
            modules.append(nonlinearity)
        
        # This line creates the relation between last hidden layer and output layer.
        modules.append(torch.nn.Linear(hidden_dimension[-1], output_dimension))
        
        # All the lines are sequentially called to create the NN.
        model = torch.nn.Sequential(*modules)
        print(model)  # Just to see the model
        return model
    
    # To predict the value based on given weights and biases
    def predict(self,x):
        return self.model(x)
    
    # Create a cost function evaluation method.
    # f is training or validation dataset, 'f_pred' predicts value from the network.
    # we can use MSEloss function of torch
    def costFunction(self, f, f_pred):
        return torch.mean((f - f_pred)**2) 
    
    # Training the neural network. Estimate the weights and biases of NN 
    def train(self, x, x_val, f, f_val, epochs, **kwargs):
        
        # Select optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), **kwargs)
        
        # Initialize history arrays. Save the values to plot wrt epochs
        self.train_cost_history = np.zeros(epochs)
        self.val_cost_history = np.zeros(epochs)
        
        # Training loop
        for epoch in range(epochs):
            f_pred = self.predict(x)
            cost = self.costFunction(f, f_pred)
            
            f_val_pred = self.predict(x_val)
            cost_val = self.costFunction(f_val, f_val_pred)
            
            self.train_cost_history[epoch] = cost
            self.val_cost_history[epoch] = cost_val
            
            # Set gradients to zero.
            self.optimizer.zero_grad()
            
            # Compute gradient (backwardpropagation). Basically find the derivative of cost function wrt to weights and biases.
            cost.backward(retain_graph=True)
            
            # Update parameters. In previous step it calculate the weights and biases but did not update.
            w = self.optimizer.step()
            
            if epoch % 1000 == 0:
                #print("Cost function: " + cost.detach().numpy())
                print(f'Epoch: {epoch}, Cost: {cost.detach().numpy()}')
            
            #plt.scatter(epoch,cost.detach().numpy()) # real time plot
            
        #plt.show()
        return epoch, cost.detach().numpy()
    
    # Plot the cost function history vs epochs as we did in earlier labs.
    def plotTrainingHistory(self, yscale='log'):
        """Plot the training history."""
        
        # Set up plot
        fig, ax = plt.subplots(figsize=(4,3))
        ax.set_title("Cost function history")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Cost function $C$")
        plt.yscale(yscale)
        
        # Plot data
        ax.plot(self.train_cost_history, 'k-', label="training cost")
        ax.plot(self.val_cost_history, 'r--', label="validation cost")
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
        plt.tight_layout()
        plt.savefig('cost-function-history.png')
        plt.show()

        
        
        
        
