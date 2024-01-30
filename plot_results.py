import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.utils import join, my_load
import os
import seaborn as sns
import numpy as np

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = 30
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['mathtext.fontset'] = 'stix'

def get_results(root_exp='experiments',
                exp_category='classification_id',
                robustness_level='semi_robust',
                dataset='imagenet',
                uq_method='None',
                backbone='resnet50',
                robust_method='wong2020',
                atk_type='O-atk',
                atk_name='Stab'):
    path = join(root_exp, exp_category, robustness_level, dataset, backbone, uq_method)
    if (robustness_level == 'semi_robust') and (robust_method is not None):
        path = join(path, robust_method)
    clean_results_path = join(path, 'clean_results.pkl')
    path = join(path, atk_type, atk_name)

    eps_to_adv_results_dict = {0: my_load(clean_results_path)}
    for eps_dir in os.listdir(path):
        eps = round(float(eps_dir.split('epsilon-')[1].split('___norm')[0])*255)
        adv_results_path_i = join(path, eps_dir, 'adv_results.pkl')

        eps_to_adv_results_dict[eps] = my_load(adv_results_path_i)

    return eps_to_adv_results_dict


if __name__ == '__main__':
    root_exp = 'experiments'
    exp_category = 'classification_id'
    robustness_level = 'semi_robust'
    dataset = 'imagenet'
    uq_method = 'None'
    backbone = 'resnet50'
    robust_method = 'wong2020'
    atk_type = 'O-atk'
    atk_name = 'Stab'

    metric = 'entropy_of_mean'

    eps_to_adv_results_dict = get_results(root_exp, exp_category, robustness_level, dataset,
                                          uq_method, backbone, robust_method, atk_type, atk_name)

    eps_list = list(eps_to_adv_results_dict.keys())


    figsize = (10, 10)  # height x width
    nrows, ncols = 1, 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*figsize[1], nrows*figsize[0]), squeeze=False)

    eps = 4

    metric_clean = eps_to_adv_results_dict[0][metric]
    metric_adv = eps_to_adv_results_dict[eps][metric]

    ax = axs[0, 0]
    ax.scatter(metric_clean, metric_adv, alpha=0.1)
    ax.set_xlabel('Entropy before attack')
    ax.set_ylabel('Entropy after attack')

    ax = axs[0, 1]
    bins = 100
    alpha = 0.5
    ax.hist(metric_clean, bins=bins, alpha=alpha, label='clean', density=True)
    ax.hist(metric_adv, bins=bins, alpha=alpha, label='adv', density=True)
    # args = {'fill': True, }
    # sns.kdeplot(metric_clean, alpha=alpha, label='clean')
    # sns.kdeplot(metric_adv, alpha=alpha, label='adv')
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Count')

    ax.legend()

    fig.show()



    print("")








