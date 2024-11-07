import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix , classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

from config import config
from optimizer import *
from utils import *
from loss import *



X, y = spiral_data(samples=config.train_samples, classes=config.num_classes)
X_val, y_val = spiral_data(samples=config.validation_samples, classes=config.num_classes)


########################## MODEL #######################################
dense1 = Layer_Dense(config.input_dim, config.hidden_1_dim, weight_regularizer_l2=config.weight_regularizer_l2, bias_regularizer_l2=config.bias_regularizer_l2)
activation1 = Activation_ReLU()

dropout1 = Layer_Dropout(config.dropout_rate)

dense2 = Layer_Dense(config.hidden_1_dim, config.num_classes)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_Adam(learning_rate=config.learning_rate, decay=config.decay)

#############################################################################



########################## Training  #######################################
loss_history = []
train_acc_history = []
val_acc_history = []
lr_history = []

for epoch in range(config.epochs+1):

    # Perform a forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)

    dropout1.forward(activation1.output)

    dense2.forward(dropout1.output)
    data_loss = loss_activation.forward(dense2.output, y)


    # Calculate regularization penalty
    regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)
    loss = data_loss + regularization_loss


    # Calculate accuracy
    predictions = np.argmax(loss_activation.output, axis=1)

    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    train_accuracy = np.mean(predictions == y)


    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)

    dropout1.backward(dense2.dinputs)

    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)


    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()


    # Log training progress
    loss_history.append(loss)
    train_acc_history.append(train_accuracy)
    lr_history.append(optimizer.current_learning_rate)


    if not epoch % config.log_step:

        # Evaluate on validation set
        dense1.forward(X_val)
        activation1.forward(dense1.output)

        dropout1.forward(activation1.output)

        dense2.forward(dropout1.output)

        val_loss = loss_activation.forward(dense2.output, y_val)
        val_predictions = np.argmax(loss_activation.output, axis=1)

        if len(y_val.shape) == 2:
            y_val = np.argmax(y_val, axis=1)

        val_accuracy = np.mean(val_predictions == y_val)
        val_acc_history.append(val_accuracy)

        print(f'epoch: {epoch}, ' +
              f'acc: {train_accuracy:.3f}, ' +
              f'val_acc: {val_accuracy:.3f}, ' +
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')
        
#############################################################################



########################## Test results  #######################################

X_test, y_test = spiral_data(samples=config.test_samples, classes=config.num_classes)


dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y_test)


predictions = np.argmax(loss_activation.output, axis=1)

if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)


accuracy = np.mean(predictions == y_test)
print(f'Test, acc: {accuracy:.3f}, Test loss: {loss:.3f}')


y_pred = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Test Confusion Matrix')
plt.show()

#############################################################################