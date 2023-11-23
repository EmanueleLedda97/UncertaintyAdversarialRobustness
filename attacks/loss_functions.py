from typing import Union
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import metrics

# In teoria la visualizzazione va fatta da altre parti ma vabbè, comodo
import pandas as pd
import matplotlib.pyplot as plt

class BaseLoss(nn.Module):
    """
    This class is just used to save loss information during training
    """
    def __init__(self, keep_loss_path=True):
        super(BaseLoss, self).__init__()
        self.keep_loss_path = keep_loss_path
        self.loss_path = {}

    def _add_loss_term(self, key: Union[str, tuple]):
        if isinstance(key, str):
            self.loss_path[key] = []
        else:
            self.loss_path['tot'] = []
            for k in key:
                self.loss_path[k] = []
                
    def _update_loss_path(self, losses, keys):
        for key, loss in list(zip(keys, losses)):
            self.loss_path[key].append(loss.item())
    
    def update_loss_path(self, loss):
        self._update_loss_path((loss,), self.loss_keys)


    def plot_loss(self, ax=None, window=20, fig_path=None):
        # todo: usare una funzione presa da un qualche visualization.py
        loss_df = pd.DataFrame(self.loss_path)

        if loss_df.shape[0] < window*10:
            window = 1
        
        if isinstance(window, int):
            loss_df = loss_df.rolling(window).mean()

        if ax is None:
            fig, ax = plt.subplots()
        loss_df.plot(ax=ax)
        ax.legend(fontsize=15)
        ax.set_xlabel('iterations')
        
        if fig_path is not None:
            fig.savefig(fig_path)


class MyCrossEntropyLoss(BaseLoss, nn.CrossEntropyLoss):
    def __init__(self):
        super(MyCrossEntropyLoss, self).__init__()
        key = 'ce'
        self._add_loss_term(key)
        self.loss_keys = tuple(self.loss_path.keys())

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)
        # Loss is returned negative so that during attack, if minimised, the error increase
        loss = - F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)
        if self.keep_loss_path:
            self.update_loss_path(loss)
        return loss


# Here I am trying to construct the Loss for the DUQ method
class RBFLoss(BaseLoss):
    def __init__(self, keep_loss_path=True):
        super().__init__(keep_loss_path)
        self.loss_ce_fn = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        # NOTE: I tried several attacks
        '''
                Cross-Entropy Attack 
            Remember, the input represent a batch containing al the similarities between
            the sample and the class. Input => [batch, similarities] (50, 10)
            The idea consists in pushing the input close to a targeted 'wrong' class.
            If I give a wrong label to maximise the similarity I obtain high confidence
            on a specific wrong class (complete evasion).
            For doing this, i pass (on main attack) not the true 'target=y', but a class shift
            by doing 'target=(y+1)%10'. By doing so, every target is a wrong class.
        '''

        loss = torch.mean(torch.tensor(1.0) - input[F.one_hot(target).bool()])
        # loss = - torch.mean(input[F.one_hot(target).bool()])
        # loss = 0.1 * self.loss_ce_fn(input, target)

        print(loss.item())
        return loss

        
class PredictionUncertaintyLoss(BaseLoss):
    def __init__(self, pred_w=1, unc_w=1):
        super(PredictionUncertaintyLoss, self).__init__()
        keys = ('pred', 'unc')
        self._add_loss_term(keys)
        self.loss_keys = tuple(self.loss_path.keys())
        self.loss_ce_fn = nn.CrossEntropyLoss()
        self.loss_ce_fn_unreduced = nn.CrossEntropyLoss(reduction='none')
        self.pred_w = pred_w
        self.unc_w = unc_w
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input: Tensor, target: Tensor, targeted=False) -> Tensor:

        # todo: ricontrollare perchè, anche con peso su unc a 0, con la MI come loss rovinava l'ottimizzazione
        # unc_loss = metrics.mutual_information(input).mean()

        mean_outs = metrics.mc_samples_mean(input)


        pred_loss = - self.loss_ce_fn(mean_outs, target.long())

        # NOTE: WARNING!!! STIAMO METTENDO UN ATTACCO DIVERSO NON CON VAR!!!
        unc_loss = metrics.mc_samples_var(input).mean()
        unc_loss = torch.log(unc_loss)

        # unc_loss = metrics.entropy(input, aleatoric_mode=False).mean()
        # unc_loss = F.cross_entropy(mean_outs, torch.argmax(mean_outs, dim=1))


        # NOTE: REMOVE THIS
        if targeted:
            pred_loss = - pred_loss

        loss = ((self.pred_w * pred_loss + self.unc_w * unc_loss) * 2) / (self.pred_w + self.unc_w)
        
        if self.keep_loss_path:
            self._update_loss_path((loss, pred_loss, unc_loss), self.loss_keys)
        return loss



