#Batches are useful implementations because
#1. The batches allow us to run multiple inputs at the same time, hence the algorithm learns more generally from all inputs
#2. It prevents from overfitting from the data
#3. The larger the batch, the more simple calculations that a GPU can do among its thousands of cores

import numpy as np

X=[[1,2,3,2.5],[2,5,1,2],[-1.5,2.7,3.3,-0.8]]

weights=[]
#bias=
output=[]
# output=np.dot(weights.T, inputs)+ bias
np.random.seed(0)

#We are creating an object to create weights between -1 and 1, the biases as an array and the output for each layer we pass forward
#We also create a biases 1 dimensional array which is as long as the no of outputs
class Layer_Dense:
    #This initializes the weights and the biases
    def __init__(self, n_inputs, n_neurons):
        # Number of neurons you want
        self.weights=0.1*(np.random.randn(n_inputs, n_neurons))#The switched order of n_inputs and n_neurons eliminates the need to transpose

        self.biases=np.zeros((1,n_neurons)) #Tuple
    #This initializes the output which returns an array
    def forward(self):
        self.output = np.dot(X, self.weights) + self.biases
print(np.random.randn(4,3))

#We will use the rectified linear activation function to model the relationships in the inputs
#This ReLU function is only activated when the input is greater than 0

class Activation_function:
    def forward (self, X):
        self.output=for i in X:
            if i > 0:
                output.append(i)
            elif i >= 0:
                output.append(0)
        return self.output

layer1=Layer_Dense(3,4)#Takes data from 3 input nodes and relays it to 4 neurons
layer2=Layer_Dense(4,5)#Takes from the previous 4 nodes, and outputs to 5 neurons

activation_layer1=Activation_function()
layer1.forward(X)#Running the inputs through the first forward propagation
activation_layer1.forward(layer1.output)#This takes out all of the negative values from the output by making them 0
print(activation_layer1.output)


#Back Propagation
#This is the algorithm that determines how much a single training example would like to alter the weights and biases for accurate output
#In essence we want to determine the alterations that would lead to the most rapid gradient descent
#Not that the implementation of this optimization may be hindered by local minima
#Now we can implement an appropriate loss function depending on our data (outliers), time efficiency of gradient descent and the confidence of predictions

#Log Loss function/cross entropy

def log_loss(X,y): # y is the labels that the data will be fitted to
    m=y.shape[0]
    p=softmax(X)
    log_likelihood = -np.long(p[range(m),y])
    loss=np.sum(log_likelihood)/m
    return loss

#Remember that we will update our weights and biases based on the negative gradient of
#the cost function in order to minimize the cost which is an array of each change that should be made for each outut
#in order to have a reliable classification
#We can quicken this up with an optimization technique.
#We can shuffle the data and split it into tiny batches to calculate the costs of each batch
#Therefore, each negative gradient will be a step in the right direction
#This would be an implementation of a greedy algorithm, hence you may jeopardise your accuracy, hence warranting many random shuffles
#You may also be interested in a simulated annealing implementation



