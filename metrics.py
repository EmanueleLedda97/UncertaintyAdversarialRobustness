import torch
import matplotlib.pyplot as plt
from utils import *
import torch.nn.functional as F
import numpy as np
import os


#  WARNING: Occhio qua che sono da controllare, ci sono alcune implementazioni che vanno
#           utilizzate e altre che invece non vanno utilizzate

'''
    All the metrics use as input a tensor of 'predictions' with size [S, B, C]
    - S -> The Monte-Carlo Sample Size
    - B -> The Batch Size
    - C -> The number of Classes

    Only the entropy accept a second parameter, which is 'aleatoric_mode'. 
    - True -> Returns the entropy of the mean prediction (which is a measure of aleatoric uncertainty)
    - False -> Returns mean of the Monte-Carlo sample individual entropies (which is a measure of epistemic uncertainty)
'''

# Expected input tensor of shape [S, B]
def get_prediction_with_uncertainty(predictions):
    return mc_samples_mean(predictions), mutual_information(predictions)

# Class-wise shortcut metrics
mc_samples_mean = lambda predictions : torch.mean(predictions, dim=0)
mc_samples_var = lambda predictions : torch.var(predictions, dim=0)
mc_samples_std = lambda predictions : torch.std(predictions, dim=0)

# Point-wise shortcut metrics
var = lambda predictions : torch.mean(mc_samples_var(predictions), dim=1)
std = lambda predictions : torch.mean(mc_samples_std(predictions), dim=1)
mutual_information = lambda predictions : entropy_of_mean_prediction(predictions) - mean_of_sample_entropies(predictions)
entropy = lambda predictions, aleatoric_mode : entropy_of_mean_prediction(predictions) if aleatoric_mode else mean_of_sample_entropies(predictions)

def true_var(predictions):
    # predictions [S, B, C]
    mean_of_square = (predictions * predictions).sum(dim=-1).mean(dim=0)
    mean = mc_samples_mean(predictions)
    square_of_mean = (mean * mean).sum(dim=1)
    var = mean_of_square - square_of_mean
    return var
'''
    Support functions
'''

# [S,B,C] => [S,B]
def __mc_sample_entropies(predictions):
    out = predictions * torch.log(predictions)          # [S,B,C] => [S,B,C]
    # out = torch.where(~torch.isnan(out), out, 0)        # [S,B,C] => [S,B,C]   
    out = torch.where(~torch.isnan(out), out, torch.tensor(0.0, dtype=torch.float32).to(out.device))         
    out = -torch.sum(out, dim=2)                        # [S,B,C] => [S,B]
    return out


# Maps [S,B,C] => [B]
def mean_of_sample_entropies(predictions):
    out = __mc_sample_entropies(predictions)            # [S,B,C] => [S,B]
    out = torch.mean(out, dim=0)                        # [S,B] => [B]
    return out


# Maps [S,B,C] => [B]
def entropy_of_mean_prediction(predictions):
    out = mc_samples_mean(predictions)                  # [S,B,C] => [B,C]
    out = mean_of_sample_entropies(out.unsqueeze(0))    # [B,C] => [1,B,C] => [B]
    return out



# Takes as input tensors of predictions, associated uncertainties and ground truths; returns the ECE
def expected_calibration_error(pred, unc, gt,
                               plot_rel_diagram=True,
                               bins=10,
                               path=os.path.join('results', 'calibration', 'ece.png')):

    
    # Defining the probability interval m and the total size
    m = 1/bins
    total_size = gt.shape[0]
    
    # Computing the accuracy inside each bucket
    ece = 0
    expected_probability, observed_probability = [], []
    for i in range(bins):
        # Getting the indices
        mask = torch.logical_and(unc >= m*i, unc < m*(i+1))
        curr_pred, curr_gt = pred[mask], gt[mask]
        bucket_size =  torch.sum(mask)

        # Computing the correct predictions inside the bucket
        correct = curr_pred.eq(curr_gt.view_as(curr_pred)).sum().item()

        # Computing the expected and observed probabilities
        curr_expected_probability = 1 - m * i
        curr_observed_probability = correct / bucket_size
        expected_probability.append(curr_expected_probability)
        observed_probability.append(curr_observed_probability)
        
        # Updating the ece only if the bucket is not empty
        if bucket_size > 0:
            ece += (bucket_size / total_size) * abs(curr_expected_probability - curr_observed_probability)

    # If required, plots the reliability diagram and save it on the pre-defined folder
    if plot_rel_diagram:
        plt.plot(expected_probability[::-1], '--', label='perfect calibration')
        plt.plot(observed_probability[::-1], 'x', label='model calibration')
        plt.savefig(path)
        plt.clf()

    return ece
    

# Utility function for computing the accuracy of a simple set
def accuracy(output, target, temperature=1):
    proba = from_logits_to_probs(output, temperature)
    pred, unc = get_prediction_with_uncertainty(proba)
    pred = pred.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct/output.shape[1]





#

