import torch
from torch.nn import Module, Softmax
from typing import Optional

# started from https://gist.github.com/SupreethRao99/2e85884dad433a6b381f966fea7a6658

class WeightedKappaLoss(Module):
    """
    Implements Quadratic Weighted Kappa Loss. Weighted Kappa Loss was introduced in the
    [Weighted kappa loss function for multi-class classification
      of ordinal data in deep learning]
      (https://www.sciencedirect.com/science/article/abs/pii/S0167865517301666).
    Weighted Kappa is widely used in Ordinal Classification Problems. The loss
    value lies in $[-\infty, \log 2]$, where $\log 2$ means the random prediction
    Usage: loss_fn = WeightedKappaLoss(num_classes = NUM_CLASSES)
    """

    def __init__(
            self,
            num_classes: int,
            device : Optional[str]     = 'cpu',
            # mode: Optional[str]        = 'quadratic',
            name: Optional[str]        = 'cohen_kappa_loss',
            epsilon: Optional[float]   = 1e-10,
            regression: Optional[bool] = True
            ):
        """Creates a `WeightedKappaLoss` instance.
            Args:
              num_classes: Number of unique classes in your dataset.
              device: (Optional) Device on which computation will be performed.
              name: (Optional) String name of the metric instance.
              epsilon: (Optional) increment to avoid log zero,
                so the loss will be $ \log(1 - k + \epsilon) $, where $ k $ lies
                in $ [-1, 1] $. Defaults to 1e-10.
              regression: (Optional) if True (default) will calculate the Loss in 
                a regression setting $ y \in R^n $, where $ n $ is the number of samples. 
                Otherwise it will assume a classification setting in which $ y \in R^{n \times m} $,
                where $ m $ is the number of classes.
            """

        super(WeightedKappaLoss, self).__init__()
        self.num_classes = num_classes

        self.epsilon = epsilon

        # Creates weight matrix (which is constant)
        self.weights = torch.Tensor(list(range(num_classes))).unsqueeze(1).repeat((1, num_classes)).to(device)
        self.weights = torch.square((self.weights - self.weights.T))

        # bricks for later histogram of values
        self.hist_bricks = torch.eye(num_classes).to(device)

        if not regression:
            self.softmax = Softmax(dim=1)
        self.regression = regression

    def kappa_loss(self, y_pred, y_true):
        num_classes = self.num_classes
        bsize = y_true.size(0)
        
        # Numerator: 
        if not self.regression:
            c = self.weights[y_true].squeeze()
            O = torch.mul(y_pred, c).sum()
        else:
            O = (y_pred - y_true).square().sum()
            
        # Denominator: 
        hist_true = torch.sum(self.hist_bricks[y_true], 0)
        
        if not self.regression: 
            hist_pred = y_pred.sum(axis=0)
        else:
            y_pred = y_pred.clamp(0, self.num_classes-1)
            y_pred_floor = y_pred.floor().long()
            y_pred_ceil  = y_pred.ceil().long()
            y_pred_perc  = (y_pred % 1).transpose(0,1)

            floor_loss = torch.mm(1-y_pred_perc, self.hist_bricks[y_pred_floor].squeeze())
            ceil_loss  = torch.mm(y_pred_perc,   self.hist_bricks[y_pred_ceil].squeeze())
            hist_pred = floor_loss + ceil_loss
            
        expected_probs = torch.mm(
            torch.reshape(hist_true, [num_classes, 1]),
            torch.reshape(hist_pred, [1, num_classes]))

        E = torch.sum(self.weights * expected_probs / bsize)

        return O / (E + self.epsilon)

    def forward(self, y_pred, y_true, log=True):
        if not self.regression:
            y_pred = self.softmax(y_pred)
        y_true = y_true.long()
        
        loss = self.kappa_loss(y_pred, y_true)
        
        if log:
            loss = torch.log(loss)
        return loss
