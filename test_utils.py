import torch

def temperature_scaling(logits, temperature):
    return torch.div(logits, temperature)