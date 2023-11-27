import torch
from torch.linalg import norm
from time import time

from attacks import constants as keys
from . import constants as keys
from . import loss_functions
from . import update_functions
# Queste sono import assolute presumendo che gli script partano dalla root
from . import constants as keys
from utils.utils import get_device
import matplotlib.pyplot as plt
from metrics import var, entropy

DEVICE = get_device()
EPSILON = keys.EPSILON

"""
    This class just define:
    - run: the general attack framework
    - compute_loss: with the classic CE, but it is overrided for the MC dropout case
    - update the input using the gradients wrt the computed loss (to be extended)
"""
class BaseAttack:
    
    def __init__(self, model, device=DEVICE, epsilon=keys.EPSILON, update_strategy='pgd', step_size=None) -> None:

        # Setting up the base attack parameters
        self.device = device
        self.model = model
        self.epsilon = epsilon

        # Choosing a suitable optimizer based on the selected update strategy
        if update_strategy == 'pgd':
            self.step_size = step_size
            self.optimizer = update_functions.PGDUpdateAndProject(epsilon=epsilon, step_size=self.step_size)
        elif update_strategy == 'fgsm':
            self.optimizer = update_functions.FGSMUpdateAndProject(epsilon=epsilon)
        else:
            raise Exception(update_strategy, "is not a supported update strategy")
        
        # Defining some parameters for managing the seceval history
        self.loss_fn = None
        self.loss = None
    
    def run(self, x, target, iterations=1):
        self.x, self.target = x, target
        self.x_adv = torch.clone(x)

        self.model.eval()
        start = time()
        for i in range(iterations):
            self.x_adv.requires_grad = True
            loss = self.compute_loss()
            loss.backward()
            self.update_and_project()
            # self.x_adv = self.x_adv.detach()          # NOTE: ho messo il detach() sotto in update_and_project() 
        end = time()
        
        self.elapsed_time = (end - start)
        # print(f"Attack Took {self.elapsed_time:.3f} seconds ({self.elapsed_time/iterations:.3f} per iter)")

        return self.x_adv.detach()
    
    def init_loss(self):
        self.loss_fn = loss_functions.MyCrossEntropyLoss()
    
    def compute_loss(self):
        output = self.model(self.x_adv)
        loss = self.loss_fn(output, self.target.long())
        return loss
    
    def update_and_project(self):
        self.x_adv = self.optimizer._update_and_project(self.x_adv, self.x).detach()


# TODO: Check if the implementation works, because maybe the ".long()" in compute_loss function generates errors
class DUQAttack(BaseAttack):
    def __init__(self, model, device=DEVICE, epsilon=keys.EPSILON, update_strategy='pgd', step_size=None) -> None:
        super().__init__(model, device, epsilon, update_strategy, step_size)
        
    def init_loss(self):
        self.loss_fn = loss_functions.RBFLoss()


