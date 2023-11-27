import torch
from . import constants as keys

class BaseUpdateAndProject:
    def __init__(self, epsilon=keys.EPSILON) -> None:
        self.epsilon = epsilon
    
    def _update_and_project(self, x_adv, x):
        raise NotImplementedError("You have to implement it by writing a specific class")


'''
    TODO: Check if it is ok
'''
class FGSMUpdateAndProject(BaseUpdateAndProject):

    def __init__(self, epsilon) -> None:
        super().__init__(epsilon)
        self.gradients = []
    
    def _update_and_project(self, x_adv, x):
        # Update
        gradient = x_adv.grad.data
        self.gradients.append(gradient.abs().max().item())
        delta = (torch.rand(x_adv.shape)*2*self.epsilon - self.epsilon).to(x_adv.device)
        delta += 1.25 * self.epsilon * gradient.sign()
        x_adv = x_adv - delta
        
        # Project
        perturb = x_adv - x
        perturb = torch.clamp(perturb, -self.epsilon, self.epsilon)
        x_adv = x + perturb
        x_adv = torch.clamp(x_adv, 0, 1)  # Adding clipping to maintain [0,1] range
        return x_adv


'''
    TODO: Add comments and documentation
'''
class PGDUpdateAndProject(BaseUpdateAndProject):

    def __init__(self, epsilon=keys.EPSILON, step_size=1e-1) -> None:
        super().__init__(epsilon)
        self.gradients = []
        self.step_size = step_size
    
    def _update_and_project(self, x_adv, x):
        # Update
        gradient = x_adv.grad.data
        self.gradients.append(gradient.abs().max().item())
        gradient = gradient.sign()
        x_adv = x_adv - self.step_size * gradient
        
        # Project
        perturb = x_adv - x
        perturb = torch.clamp(perturb, -self.epsilon, self.epsilon)

        x_adv = x + perturb
        x_adv = torch.clamp(x_adv, 0, 1)  # Adding clipping to maintain [0,1] range
        return x_adv

