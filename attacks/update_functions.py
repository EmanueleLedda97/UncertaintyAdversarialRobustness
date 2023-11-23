import torch
from . import constants as keys

class BaseUpdateAndProject:
    def __init__(self, 
                 epsilon=keys.EPSILON, 
                 step_size=None) -> None:
        self.epsilon = epsilon
        self.step_size = step_size
    
    def _update_and_project(self, x_adv, x):
        raise NotImplementedError("You have to implement it by writing a specific class")


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


# TODO: Aggiungere il sign
# TODO: Verificare che la epsilon sia raggiunto
# TODO: Usare norma inf
class PGDUpdateAndProject(BaseUpdateAndProject):
    def __init__(self, epsilon=keys.EPSILON, step_size=1e-1) -> None:
        super().__init__(epsilon, step_size)
        self.gradients = []
    
    def _update_and_project(self, x_adv, x):
        # Update
        gradient = x_adv.grad.data
        self.gradients.append(gradient.abs().max().item())
        # gradient /= gradient.norm(p=2)
        gradient = gradient.sign()
        x_adv = x_adv - self.step_size * gradient
        
        
        # Project
        perturb = x_adv - x
        perturb = torch.clamp(perturb, -self.epsilon, self.epsilon)

        # TODO: ACCROCCHIO PER LA PATCH
        # w, h = 520, 736
        # w_m, h_m = w//2, h//2
        # p_m = 75
        # mask = torch.zeros(size=(1, 3, w, h))
        # mask[:, :, (w_m-p_m):(w_m+p_m), (h_m-p_m):(h_m+p_m)] = 1
        # perturb = perturb * mask.to('cuda:1')

        x_adv = x + perturb
        x_adv = torch.clamp(x_adv, 0, 1)  # Adding clipping to maintain [0,1] range
        return x_adv

# class PGDUpdateAndProject(BaseUpdateAndProject):
#     """
#     TODO NON C'è IMPLEMENTATO NULLA ANCORA ED è DA AGGIORNARE
#     """
    
#     def __init__(self, **kwargs) -> None:
#         super().__init__(**kwargs)
        
#     def update_and_project(self):
#         # Update
#         gradient = self.x_adv.grad.data
#         gradient /= gradient.norm(p=2)
#         self.x_adv = self.x_adv + self.step_size * gradient
        
#         # Project
#         perturb = self.x_adv - self.x
#         perturb = torch.clamp(perturb, -self.epsilon, self.epsilon)
#         self.x_adv = self.x + perturb
#         self.x_adv = torch.clamp(self.x_adv, 0, 1)  # Adding clipping to maintain [0,1] range

#         return