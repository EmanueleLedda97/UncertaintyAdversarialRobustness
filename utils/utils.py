import torch
import pickle
import numpy as np
import random
import os

import torchvision
import utils.constants as keys
from torchvision import transforms
from datetime import datetime
import json



'''
    NOTE: This file still needs to be refactored!
'''



##################################################################################
# TRAINING CHECKPOINTS
##################################################################################

def extract_checkpoints(root_path):
    for f in os.listdir(root_path):
        if 'ckpt' in f:
            model_file_name = f.replace('checkpoint', 'model')
            # model_file_name = f[:-4] + 'pt'
            model_file_name = model_file_name.replace('ckpt', 'pt')
            ckpt = torch.load(os.path.join(root_path, f))
            stripped_ckpt_state_dict = strip_checkpoint(ckpt)
            torch.save(stripped_ckpt_state_dict, os.path.join(root_path, model_file_name))


def strip_checkpoint(checkpoint):
    state_dict = checkpoint['state_dict']
    keys = state_dict.copy().keys()
    for k in keys:
        stripped_k = k[6:]  # Removing the "model." string
        state_dict[stripped_k] = state_dict.pop(k)
    
    return state_dict


def extract_all_existing_checkpoints(folders=['embedded_dropout', 'deep_ensemble']):
    for method in folders:    
        for resn in ['resnet18', 'resnet34', 'resnet50']:
            r = os.path.join('models', method, resn)
            extract_checkpoints(r)


##################################################################################
# DATA UTILS
##################################################################################

# get the preprocessing layers for a given dataset
def get_normalizer(dataset='cifar10'):

    # Obtaining mean and variance for the data normalization
    mean, std = keys.NORMALIZATION_DICT[dataset]
    normalizer = transforms.Normalize(mean=mean, std=std)

    return normalizer


# Function for obtaining a unified division between training, test and validation
def get_dataset_splits(dataset='cifar10', set_normalization=True, ood=False, load_adversarial_set=False, num_advx=None):
    set_all_seed(keys.DATA_SEED)

    # TODO: Find the way for obtaining a ood version
    # Deterioration transformation used for OOD
    deterioration_preprocess = [transforms.ElasticTransform(alpha=250.0), transforms.AugMix(severity=10), transforms.ToTensor()]

    # Defining the validation data-preprocesser
    val_preprocess = [transforms.ToTensor()]

    # Defining the training data-preprocesser with data augmentation
    train_preprocess = [
        # transforms.RandomCrop(32, padding=4), # REMOVED RANDOM CROP
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
    
    if set_normalization:
        normalizer = get_normalizer(dataset)
        train_preprocess.append(normalizer)
        val_preprocess.append(normalizer)
    
    train_preprocess = transforms.Compose(train_preprocess)
    val_preprocess = transforms.Compose(val_preprocess)

    # if corrupted:
    #     val_preprocess = transforms.Compose(deterioration_preprocess)

    if ood:
        dataset = 'cifar100'

    # Loading the original sets
    if dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root='datasets', train=True, download=True, transform=train_preprocess)
        test_set = torchvision.datasets.CIFAR10(root='datasets', train=False, download=True, transform=val_preprocess)
    # Loading the original sets
    elif dataset == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(root='datasets', train=True, download=True, transform=train_preprocess)
        test_set = torchvision.datasets.CIFAR100(root='datasets', train=False, download=True, transform=val_preprocess)
    else:
        raise Exception(f'{dataset} is not supported at the moment. Try using cifar10.')

    # Using 8000 images for test and 2000 for validation
    test_set, validation_set = torch.utils.data.random_split(test_set, [8000, 2000])

    if load_adversarial_set:
        set_all_seed(keys.DATA_SEED)
        adversarial_test_set, _ = torch.utils.data.random_split(test_set, [num_advx, len(test_set) - num_advx])
        return adversarial_test_set

    return train_set, validation_set, test_set


class AdversarialDataset(torch.utils.data.Dataset):
    """
    This retrieve the saved adversarial examples in the form of <sample_id>.gz
    """
    def __init__(self, ds_path, transforms=None):
        self.ds_path = ds_path
        self.transforms = transforms
        all_samples = os.listdir(self.ds_path)
        all_samples.remove('fname_to_target.json')
        self.n_samples = len(all_samples)
        with open(join(ds_path, 'fname_to_target.json')) as json_file:
            self.fname_to_target = json.load(json_file)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        fname = f"{str(index).zfill(10)}.png"
        file_path = os.path.join(self.ds_path, fname)
        x = torchvision.io.read_image(file_path)/255.
        y = self.fname_to_target[fname]
        if self.transforms is not None:
            x = self.transforms(x)
        return x, y


##################################################################################
# FILE MANAGEMENT 
##################################################################################

get_device = lambda id = 0 : f"cuda:{id}" if torch.cuda.is_available() else 'cpu'

# TODO: Add documentation
def my_load(path, format='rb'):
    with open(path, format) as f:
        object = pickle.load(f)
    return object


# TODO: Add documentation
def my_save(object, path, format='wb'):
    with open(path, format) as f:
        pickle.dump(object, f)


# Same as os.path.join, but it replaces all '\\' with '/' when running on Windows
def join(*args):
    path = os.path.join(*args).replace('\\', '/')
    return path


# Procedure for setting all the seeds
def set_all_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)










##################################################################################
# EXPERIMENTS PATH
##################################################################################
# TODO: to be documented
def get_base_exp_path(root, dataset, uq_technique, backbone):
    exp_path = join(root, dataset, uq_technique, backbone)
    return exp_path

# TODO: to be documented
def get_paths(root, dataset, uq_technique, dropout_rate, backbone,
              eps, atk_type='pgd-pred', step_size=1, num_advx=100, 
              pred_w=1, unc_w=1, mc_atk=30):
    
    # TODO: Why 'base_exp_path' ? 
    base_exp_path = get_base_exp_path(root, dataset, uq_technique, backbone)

    # Obtaining the basic experimental setup string
    advx_dir_name = f"eps-{eps:.3f}--step_size-{step_size:.3f}-num_advx-{num_advx}-" \
                    f"pred_w-{pred_w}-unc_w-{unc_w}-mc_atk-{mc_atk}"
    advx_dir_name = f"{advx_dir_name}-dr-{dropout_rate}" if "dropout" in uq_technique else advx_dir_name

    # advx_dir_name = get_advx_dir_name(eps, step_size, num_advx, pred_w, unc_w, mc_atk)
    advx_exp_path = join(base_exp_path, atk_type, advx_dir_name)
    advx_results_path = join(advx_exp_path, 'adv_results.pkl')
    baseline_results_path = join(base_exp_path, 'clean_results.pkl')

    # TODO: return a dictionary instead N string
    # paths = {'exp_path': exp_path,
    #          ''}

    return base_exp_path, advx_exp_path, advx_results_path, baseline_results_path

# TODO: fix the 0 arg error
# def get_advx_dir_name(**kwargs):
#     s = ''
#     for k, v in kwargs.items():
#         if isinstance(v, float):
#             v = f"{v:.3f}"
        
#         if s == '':
#             s += f"{k}-{v}"
#         else:
#             s += f"-{k}-{v}"


##################################################################################
# OTHERS
##################################################################################

# Takes as inputs a logits vector and perform temperature scaling on that vector: new_l = l/t
def temperature_scaling(logits, temperature):
    return torch.div(logits, temperature)


# Utility function for converting logits to probabilities
def from_logits_to_probs(logits, temperature):
    scaled_logits = temperature_scaling(logits, temperature)
    return F.softmax(scaled_logits, dim=-1)


# NOTE: WARNING! This function needs to be finished
# TODO: Finish and add documentation
def plot_uncertainty_distributions(output, target, temperature):

    # Obtining prediction and uncertainty with correct mask
    proba = from_logits_to_probs(output, temperature)
    pred, unc = get_prediction_with_uncertainty(proba)
    pred = pred.argmax(dim=1, keepdim=True)
    mask = pred.eq(target.view_as(pred)).view_as(unc)

    # print(mask.shape, unc.shape)
    correct_samples = unc[mask]
    incorrect_samples = unc[~mask]

    # Printing the moments of the distribution
    print(f"max unc {torch.max(unc)} - min unc {torch.min(unc)}")
    print(f"CORRECT: mean {torch.mean(correct_samples)} - var {torch.var(correct_samples)}")
    print(f"INCORRECT: mean {torch.mean(incorrect_samples)} - var {torch.var(incorrect_samples)}") 


# NOTE: WARNING! This function needs to be finished
# TODO: Finish and add documentation
def search_for_optimal_calibration(output, target, low_b=1, high_b=3, resolution=30):
    
    base_path = os.path.join('results', 'calibration', )

    # Computing the temperatures to be tested
    interval = high_b - low_b
    step = interval / resolution
    temps = np.arange(resolution) * step + low_b

    for b in [10, 25, 50]:
        eces = []
        for t in temps:
            scaled_output = temperature_scaling(output, t)
            proba = F.softmax(scaled_output, dim=-1)
            pred, unc = get_prediction_with_uncertainty(proba)
            pred = pred.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            ece = expected_calibration_error(pred, unc, target, bins=b)
            eces.append(ece)
        plt.plot(temps, eces)
        plt.savefig(os.path.join(base_path, f'eces_comparison{b}.png'))



##################################################################################
# VISUALIZATION
##################################################################################
import matplotlib.pyplot as plt

def create_legend(ax, figsize=(10, 0.5)):
    # create legend
    h, l = ax.get_legend_handles_labels()
    legend_dict = dict(zip(l, h))
    legend_fig = plt.figure(figsize=figsize)

    legend_fig.legend(legend_dict.values(), legend_dict.keys(), loc='center',
                      ncol=len(legend_dict.values()), frameon=False)
    legend_fig.tight_layout()

    return legend_fig