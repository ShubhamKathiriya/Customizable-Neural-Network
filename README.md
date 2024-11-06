# Customizable Neural Network from Scratch











## Libraries

```code
    pip install numpy nnfs matplotlib seaborn scikit-learn
```


- Numpy - for whole Neural Net implementation
- nnfs - for spiral dataset 
- matplot / seaborn - for visualization and plotting purpose
- sklearn - for confussion and classification report

default regularize -> all 0

# adam default setting 

learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999





# gradient clipping 

Training Stability:


Without clipping: Network might never recover from bad predictions
With clipping:

Provides reasonable gradients even for very wrong predictions
Allows network to learn from mistakes
Prevents training from getting stuck or exploding



Log of zero or negative numbers is undefined in real numbers
Log of numbers very close to zero approaches negative infinity
This would cause numerical instability in computations