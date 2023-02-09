# Weighted Kappa Loss
 Pytorch module for computing Quadratic Weighted Kappa Loss


Weighted Kappa Loss was introduced in the [Weighted kappa loss function for multi-class classification of ordinal data in deep learning](https://www.sciencedirect.com/science/article/abs/pii/S0167865517301666). 

Weighted Kappa is widely used in Ordinal Classification Problems. The loss value lies in $[-\infty, \log 2]$, where $\log 2$ means the random prediction

## Features
Can be adapted and used both in regression and classification environments, by regulating the attribute `regression=True|False`. 

## Usage
Copy the code by typing:

```python
!git clone https://github.com/mamiglia/WeightedKappaLoss.git
```

The module is used as any other loss function module of pytorch.

```python
from WeightedKappaLoss import WeightedKappaLoss

loss_fn = WeightedKappaLoss(
    num_classes = NUM_CLASSES, 
    device = DEVICE,
    regression = True or False
)

...

y_hat = model(X)
loss = loss_fn(y_hat, y_true)
loss.backward()
```

## Issues

Currently only the **quadratic** loss is implemented.

## Credits

Thanks [@SupreethRao99](https://github.com/SupreethRao99) for [this](https://gist.github.com/SupreethRao99/2e85884dad433a6b381f966fea7a6658) piece of code which I randomly found while browsing, and that got me started.