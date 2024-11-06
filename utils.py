import numpy as np

class Activation_ReLU:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0



class Activation_Softmax:
    
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1,keepdims=True)
        self.output = probabilities



class Layer_Dropout:

    def __init__(self, rate):
        # Store the dropout rate, invert it to get the success rate
        self.rate = 1 - rate


    def forward(self, inputs):

        self.inputs = inputs

        # Generate and save the scaled binary mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask


    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask