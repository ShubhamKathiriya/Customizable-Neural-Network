import numpy as np
from config import config
from utils import Activation_Softmax


class Loss:

    def regularization_loss(self, layer):

        regularization_loss = 0

        # L1 regularization - weights
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

        # L2 regularization - weights
        if layer.weight_regularizer_l2 > 0:

            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

        # L1 regularization - biases
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

        # L2 regularization - biases
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss



    # Calculates the data and regularization losses given model output and ground truth values
    def calculate(self, output, y):

        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    


class Loss_CategoricalCrossentropy(Loss):

    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        # Clip data to prevent division by 0 and Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, config.output_clip , 1 - config.output_clip)


        # Probabilities for target values - only if categorical labels        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[ range(samples),y_true]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum( y_pred_clipped * y_true,axis=1)

        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])


        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues

        # Normalize gradient
        self.dinputs = self.dinputs / samples






# Softmax classifier - combined Softmax activation and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy:

    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):

        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.calculate(self.output, y_true)

    
    def backward(self, dvalues, y_true):
        
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1

        # Normalize gradient
        self.dinputs = self.dinputs / samples
