from attacks import constants as keys
from attacks.standard_attacks import DEVICE, keys
from .standard_attacks import *
# Queste sono import assolute presumendo che gli script partano dalla root
import metrics


class BaseBayesianAttack(BaseAttack):
    """
    This class have the bayesian loss that performs EOT against MC Dropout
    kwargs contains epsilon and step_size of the BaseAttack
    """
    def __init__(self, mc_sample_size_during_attack=20, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mc_sample_size_during_attack = mc_sample_size_during_attack

    def init_loss(self):
        self.loss_fn = loss_functions.BayesianCrossEntropyLoss()
    
    def compute_loss(self):
        output = self.model(self.x_adv, 
                            mc_sample_size=self.mc_sample_size_during_attack,
                            get_mc_output=True)
        loss = self.loss_fn(output, self.target)
        return loss
        

class StabilizingAttack(BaseBayesianAttack):
    def __init__(self, mc_sample_size_during_attack=20, **kwargs) -> None:
        super().__init__(mc_sample_size_during_attack, **kwargs)
    
    def init_loss(self):
        self.loss_fn = loss_functions.BayesianCrossEntropyLoss()
    
    # Get the most likely class
    def get_target_from_output(self):
        pass


class AutoTargetAttack(BaseBayesianAttack):
    def __init__(self, mc_sample_size_during_attack=20, **kwargs) -> None:
        super().__init__(mc_sample_size_during_attack, **kwargs)
    
    def init_loss(self):
        self.loss_fn = loss_functions.BayesianCrossEntropyLoss()

    # Get the most likely wrong class
    def get_target_from_output(self):
        pass


class MinVarAttack(BaseBayesianAttack):
    def __init__(self, mc_sample_size_during_attack=20, **kwargs) -> None:
        super().__init__(mc_sample_size_during_attack, **kwargs)
    
    def init_loss(self):
        self.loss_fn = loss_functions.VarianceLoss(beta=1)


class MaxVarAttack(BaseBayesianAttack):
    def __init__(self, mc_sample_size_during_attack=20, **kwargs) -> None:
        super().__init__(mc_sample_size_during_attack, **kwargs)
    
    def init_loss(self):
        self.loss_fn = loss_functions.VarianceLoss(beta=-1)
    





#####################################################################

# class FGSMBayesianAttack(BaseBayesianAttack):
#     def __init__(self, epsilon=keys.EPSILON, **kwargs) -> None:
#         super().__init__(**kwargs)
#         self.optimizer = update_functions.FGSMUpdateAndProject(epsilon=epsilon)


class PGDBayesianAttack(BaseBayesianAttack):
    def __init__(self, epsilon=keys.EPSILON, step_size=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.optimizer = update_functions.PGDUpdateAndProject(epsilon=epsilon, step_size=step_size)
        







'''
    TODO: Fix this section when refactoring the semantic segmentation part
'''
class PGDSemanticSegmentationAttack(PGDBayesianAttack):
    def __init__(self,
                 epsilon=keys.EPSILON,
                 step_size=1,
                 centered=False,
                 w=520, h=736,
                 mask_size=50,
                 targeted=False,
                 **kwargs) -> None:
        super().__init__(epsilon, step_size, **kwargs)
        self.mask = torch.zeros(size=(1, w, h))
        self.targeted = targeted
        # self.mask[:, :mask_size, :mask_size] = 1
        # self.mask = self.mask.flatten()


    def compute_loss(self, duq=False):
        # todo: modularizzare questa parte in modo che se voglio attaccare un non bayesiano lo faccio
        # output = self.model(self.x_adv, 
        #                     mc_sample_size=self.mc_sample_size_during_attack,
        #                     get_mc_output=True)
        
        output = self.model(x=self.x_adv, 
                            mc_sample_size=self.mc_sample_size_during_attack,
                            get_mc_output=True)
        


        s, b, c, w, h = output.shape    # S, B, 21, W, H  -> 100, 
        output = torch.transpose(output, 2, 4)
        output = torch.reshape(output, (s, b*h*w, c))

        # If I am using deterministic UQ I need to mantain the target unchanged (maybe?)
        loss = self.loss_fn(output, self.target.flatten().long(), targeted=self.targeted)
            
        
        # todo: fare qualcosa che raccolga la loss in un file o in un vettore
        return loss
