import torch
from torch.linalg import norm
from time import time
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

# REFACTORING-FALL: Questo script va completamente ricontrollato.
class BaseAttack:
    """
    This class just define:
    - run: the general attack framework
    - compute_loss: with the classic CE, but it is overrided for the MC dropout case
    - update the input using the gradients wrt the computed loss (to be extended)
    """
    def __init__(self) -> None:
        """
        Step size non serve in FGSM!!!
        """
        # self.epsilon = epsilon
        # self.step_size = step_size

        # TODO: gestire il caso dove l'attacco viene lanciato più volte, 
        # quindi mantenere una curva per ogni run
        self.loss_fn = None
        self.loss = None
    
    def run(self, x, target, model, iterations=1, device=DEVICE, duq=False):
        self.device = device
        self.model = model
        self.x = x
        self.target = target
        self.x_adv = torch.clone(x)
        
        # If I am using deterministic UQ I change the loss
        if duq:
            self.init_duq_loss()
        else:
            self.init_loss()

        #   --- SEMANTIC SEGMENTATION ---
        # M, m = 0, 0
        # fig, axs = plt.subplots(nrows=2,
        #                         ncols=5, 
        #                         squeeze=False,
        #                         figsize=(10,5))
        # var_list = []
        # etp_list = []
        self.model.eval()
        start = time()
        for i in range(iterations):
            # print(i)

            #   --- SEMANTIC SEGMENTATION ---
            # st = iterations//5
            
            # if i%st == 0:
                
            #     test_pred = None
            #     for k in range(1):
            #         # Use the model and visualize the prediction
            #         test_pred_curr = model(x=self.x_adv.to(device), mc_sample_size=5, get_mc_output=True).cpu().detach()

            #         test_pred = test_pred_curr if test_pred is None else torch.cat((test_pred, test_pred_curr), 0)
            #         # print(test_pred.shape[0])
                
            #     # test_pred = self.model(self.x_adv.to('cuda:1'), mc_sample_size=5, get_mc_output=True).cpu().detach()
            #     test_pred = test_pred.softmax(dim=2)
            #     # test_pred = test_pred.mean(dim=0)[0]
            #     etp_adv = entropy(test_pred, aleatoric_mode=True)[0]
            #     var_adv = var(test_pred)[0]
            #     if i==0:
            #         etp_M, etp_m = etp_adv.max().item(), etp_adv.min().item()
            #         var_M, var_m = var_adv.max().item(), var_adv.min().item()
            #     else:
            #         etp_M = max(etp_M, etp_adv.max().item())
            #         etp_m = min(etp_m, etp_adv.min().item())
            #         var_M = max(var_M, var_adv.max().item())
            #         var_m = min(var_m, var_adv.min().item())
            #     var_list.append(var_adv)
            #     etp_list.append(etp_adv)
            
            self.x_adv.requires_grad = True
            loss = self.compute_loss(duq=duq)
            loss.backward()
            self.update_and_project()
            self.x_adv = self.x_adv.detach()
            
        #   --- SEMANTIC SEGMENTATION ---
        # for j, u in enumerate(var_list):
        #     axs[0, j].set_title(f'{u.mean():.6f}')
        #     axs[0, j].imshow(u, vmax=var_M, vmin=var_m, cmap='jet')
        #     axs[0, j].axis('off')
        
        # for j, u in enumerate(etp_list):
        #     axs[1, j].set_title(f'{u.mean():.3f}')
        #     axs[1, j].imshow(u, vmax=etp_M, vmin=etp_m, cmap='jet')
        #     axs[1, j].axis('off')
        # plt.savefig(f'comparison.png')
        end = time()
        
        self.elapsed_time = (end - start)#/iterations
        # print(f"Attack Took {self.elapsed_time:.3f} seconds ({self.elapsed_time/iterations:.3f} per iter)")
        
        # print(f"TIME: {self.elapsed_time}")
        # exit(1)
        # Return the perturbed image
        return self.x_adv.detach()
    
    def init_loss(self):
        self.loss_fn = loss_functions.MyCrossEntropyLoss()
    
    def compute_loss(self):
        output = self.model(self.x_adv)
        loss = self.loss_fn(output, self.target.long())
        return loss
    
    ''' REFACTORING-FALL
        Quindi se non ho capito male questo serve per capire quale update and project fare;
        QUINDI alla fine la differenza tra i due attacchi sta semplicmenete nell'update and project
        che scegliamo, non c'è nessun altra differenza, l'unica differenza che conta è questa!
    '''
    # def update_and_project(self):
        # raise NotImplementedError("You have to implement it by writing a specific class")
    def update_and_project(self):
        # todo: questo può essere inglobato in classi parent, perchè è ripetuta praticamente uguale
        self.x_adv = self.optimizer._update_and_project(self.x_adv, self.x)
        return


class FGSMAttack(BaseAttack):
    def __init__(self, epsilon=keys.EPSILON, step_size=1) -> None:
        # step_size qui è una porcheria perchè non viene mai usata, però è per non rompere l'interfaccia generale
        super().__init__()
        self.optimizer = update_functions.FGSMUpdateAndProject(epsilon=epsilon)
    
    # def update_and_project(self):
    #     self.x_adv = self.optimizer._update_and_project(self.x_adv, self.x)
    #     return


class PGDAttack(BaseAttack):
    def __init__(self, epsilon=keys.EPSILON, step_size=1) -> None:
        super().__init__()
        self.optimizer = update_functions.PGDUpdateAndProject(epsilon=epsilon, step_size=step_size)
    
    # def update_and_project(self):
    #     self.x_adv = self.optimizer._update_and_project(self.x_adv, self.x)
    #     return


if __name__ == '__main__':
    pass