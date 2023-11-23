from attacks import constants as keys
from attacks.standard_attacks import keys
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

    def init_duq_loss(self):
        self.loss_fn = loss_functions.RBFLoss()

    def init_loss(self):
        self.loss_fn = loss_functions.PredictionUncertaintyLoss(pred_w=self.attack_pred, unc_w=self.attack_unc)
    
    def compute_loss(self, duq=False):
        output = self.model(self.x_adv, 
                            mc_sample_size=self.mc_sample_size_during_attack,
                            get_mc_output=True)
        
        # If I am using deterministic UQ I need to mantain the target unchanged (maybe?)
        if duq:
            loss = self.loss_fn(output, self.target)
        else:
            loss = self.loss_fn(output, self.target.long(), targeted=self.targeted)
            
        return loss
        
    



class FGSMBayesianAttack(BaseBayesianAttack):
    def __init__(self, epsilon=keys.EPSILON, step_size=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.optimizer = update_functions.FGSMUpdateAndProject(epsilon=epsilon)


class PGDBayesianAttack(BaseBayesianAttack):
    def __init__(self, epsilon=keys.EPSILON, step_size=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.optimizer = update_functions.PGDUpdateAndProject(epsilon=epsilon, step_size=step_size)
        


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
