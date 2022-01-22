from audioop import bias
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
X   =    [
            [0.1, 0.12, 0.383, -0.6],
            [0.2, 0.5, -0.7, 0.9],
            [-0.172,0.44,-0.67,0.89]
         ]


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs,n_neurons)
        self.weights /= np.max(self.weights)
        self.biases = np.zeros((1,n_neurons))

    def forward(self,inputs):
        self.output = inputs @ self.weights + self.biases
        return self.output
    


l0 = Layer_Dense(4,10)
l1 = Layer_Dense(10,2)
l2  = Layer_Dense(2,1)

print(l2.forward(l1.forward(l0.forward(X))))





# weights1 = [[1,2,4,6],
#            [1, 5,8,0],
#            [9,10, 11,12]]

# weights2 = [[1,0.08,4],
#            [1, 5,8.22],
#            [9,10, 0.11]]


# biases1 = [1, 2, 3]
# biases2 = [1.09, 2.12, 3.23]


# layer1_outputs = inputs @ np.array(weights1).T + biases1


# layer2_outputs = layer1_outputs @ np.array(weights2).T + biases2

# print(layer1_outputs)
# print()
# print(layer2_outputs)