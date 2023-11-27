'''
    TODO: File temporaneo di prova. Ovviamente e' da integrare altrove. sempre questione refactoring
'''


import torch

def temperature_scaling(logits, temperature):
    return torch.div(logits, temperature)