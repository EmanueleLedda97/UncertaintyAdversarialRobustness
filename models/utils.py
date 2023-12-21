from models.resnet import ResNetMCD, ResNetEnsemble, ResNet_DUQ
import utils.constants as keys
import torch.nn as nn
import torch
import utils.constants as keys
import os

# TODO: IMplementare un parametro per caricare un modello robusto con AT
# REFACTORING-FALL: TODO: Add a parameter for the semantic segmentation
'''
    name: load_model
    parameters:
        - backbone      -> The backbone network architecture
        - uq_technique  -> The chosen uncertainty quantification technique
        - dataset       -> The dataset on which the model has been trained on
        - dropout_rate  -> The chosen dropout rate for the injected/embedded dropout

    The purpose of this function is to provide a direct interface for loading the already trained
    uncertainty aware models.
'''
def load_model(backbone, uq_technique, dataset, dropout_rate=None, full_bayesian=False, ensemble_size=5, transform=None, device='cpu'):

    # Guard for the supported network architectures
    if backbone not in keys.SUPPORTED_BACKBONES:
        raise Exception(f"{backbone} is not a supported network architecture")
    
    # Guard for the supported data sets
    if dataset not in keys.SUPPORTED_DATASETS:
        raise Exception(f"{dataset} is not a supported dataset")

    # Matching the UQ Technique
    if 'dropout' in uq_technique:
        
        # Obtaining the correct dropout rate
        if dropout_rate == None:
            dropout_rate = keys.DEFAULT_DROPOUT_RATE
        elif dropout_rate not in keys.SUPPORTED_DROPOUT_RATES:
            raise Exception(f"You should select one of the following dropout rates {keys.SUPPORTED_DROPOUT_RATES}")
        
        # Loading the correct MCD ResNet
        if backbone in keys.SUPPORTED_RESNETS:     
            temperature = 1.0   # TODO: Find a more elegant solution

            # Creating the MCD ResNet
            model = ResNetMCD(backbone, pretrained=True, # default = pretrained TRUE
                              dropout_rate=dropout_rate,
                              full_bayesian=full_bayesian,
                              temperature=temperature,
                              transform=transform)

            # If the technique is embedded dropout we load the special embedded weights
            if uq_technique == 'embedded_dropout':
                embedding_type = 'embedded_dropout_full_bayes' if full_bayesian else 'embedded_dropout'
                dropout_id = int(dropout_rate * 10)
                embedded_path = os.path.join('models', embedding_type, backbone, f"model_dr{dropout_id}.pt")
                # model.backbone.load_state_dict(torch.load(embedded_path, map_location=torch.device(device)))

        elif backbone in keys.SUPPORTED_VGGS:
            raise Exception("Vgg are not implemented yet!")
        
        # Preparing the network dropout layers
        set_model_to_eval_activating_dropout(model)
    elif uq_technique == 'deep_ensemble':
        temperature = 1.0
        model = ResNetEnsemble(backbone, ensemble_size, transform=transform)
        model.eval()
    elif uq_technique == 'deterministic_uq':
        model = ResNet_DUQ(transform=transform)
        model.eval()
    else:
        raise Exception(f"{uq_technique} is not a supported uncertainty quantification technique.")

    return model
    

# Private procedure for set the network to eval while keeping active the dropout layers
def set_model_to_eval_activating_dropout(model):

    # Set to eval mode (no gradients, no batch norm)
    model.eval()

    # Reactivating the dropout layers
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.training = True

    
    