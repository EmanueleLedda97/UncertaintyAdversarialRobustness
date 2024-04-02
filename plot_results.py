
import matplotlib.pyplot as plt
from utils.utils import join, my_load
import utils.constants as const
import utils.paths as utpaths
import os
import seaborn as sns
import numpy as np
import pickle
import torch
import io
import utils.visualisation as viz
import scipy
import math
import seaborn as sns
from matplotlib.colors import TABLEAU_COLORS

from utils.constants import CIFAR10_ROBUST_MODELS, IMAGENET_ROBUST_MODELS, \
    CIFAR10_NAIVE_MODELS, IMAGENET_NAIVE_MODELS, \
    cifar10_model_dict, imagenet_model_dict
from matplotlib.lines import Line2D

COLORS = ['tab:green', 'tab:orange', 'tab:blue', 'tab:red']
ALL_COLORS = list(TABLEAU_COLORS.keys())
ALL_MARKERS = list("o*sP^<>1234vXD")


"""
PLOT FILE WITH ONLY ESSENTIAL PLOTS
"""



def get_paths(datasets, atk_type_and_name_list, robusts=[True]):
    path_ok_list = []
    for atk_type_and_name in atk_type_and_name_list:
        print('#' * 50)
        atk_type = f"Attack type: {atk_type_and_name}"
        print(f"#{atk_type.center(48, ' ')}#")
        print('#' * 50)
        for dataset in datasets:
            s = f" Dataset: {dataset} "
            print(s.center(50, '+'))
            for robust in robusts:
                s = " Robust Models " if robust else " Naive Models "
                print(s.center(50, '-'))
                base_dataset_path = join("experiments/classification_id",
                                          "semi_robust" if robust else "naive_robust",
                                          dataset)
                if dataset == 'cifar10':
                    if robust:
                        model_list = CIFAR10_ROBUST_MODELS
                        model_dict = cifar10_model_dict
                    else:
                        model_list = CIFAR10_NAIVE_MODELS
                elif dataset == 'imagenet':
                    if robust:
                        model_list = IMAGENET_ROBUST_MODELS
                        model_dict = imagenet_model_dict
                    else:
                        model_list = IMAGENET_NAIVE_MODELS
                else:
                    print("Unsupported dataset.")
                    exit(0)

                if robust:
                    paths = [join(
                        base_dataset_path, model_dict[model_name]['resnet_type'], 'None', model_name, atk_type_and_name)
                        for model_name in model_list]
                else:
                    paths = [join(base_dataset_path, model_name, 'None', atk_type_and_name) for model_name in model_list]

                path_ok_list.append(check_exp(paths))
                # try:
                #     details = f"_{dataset}_{atk_type_and_name.split('/')[1]}_{s.split()[0]}"
                #     # plot_conf_displacement(path_ok_list)
                #     plot_cal_curves(path_ok_list, figname=f"calibration_curve_{details}")
                # except:
                #     print("")
                # plot_ECE(path_ok_list, figname=f"ECE_{details}")
                print("")
            print('/' * 50)
        print("\\" * 50)
    return path_ok_list


def get_results(*args):
    if 'semi_robust' in args:
        root_exp, exp_category, robustness_level, dataset, backbone, \
            uq_method, robust_method, atk_type, atk_name, *_ = args
    elif 'naive_robust' in args:
        root_exp, exp_category, robustness_level, dataset, backbone, uq_method,\
            atk_type, atk_name, *_ = args


    path = join(root_exp, exp_category, robustness_level, dataset, backbone, uq_method)
    if (robustness_level == 'semi_robust') and (robust_method is not None):
        path = join(path, robust_method)
    clean_results_path = join(path, 'clean_results.pkl')
    path = join(path, atk_type, atk_name)

    # eps_to_adv_results_dict = {0: my_load(clean_results_path)}
    eps_to_adv_results_dict = {0: my_load(clean_results_path)}
    eps_to_adv_results_dict['not_ok_eps_path'] = []

    step_size = 0.002
    try:
        for eps_dir in os.listdir(path):
            if f"step-{step_size}" not in eps_dir:
                continue
            eps = round(float(eps_dir.split('epsilon-')[1].split('___norm')[0]) * 255)
            adv_results_path_i = join(path, eps_dir, 'adv_results.pkl')
            try:
                eps_to_adv_results_dict[eps] = my_load(adv_results_path_i)
            except:
                eps_to_adv_results_dict['not_ok_eps_path'].append(adv_results_path_i)

    except:
        eps_to_adv_results_dict['not_ok_eps_path'].append(path)


    return eps_to_adv_results_dict


def get_bucket_acc(confidences, y_true, y_preds, min_conf=.0, max_conf=.1):
    bucket_filter = (confidences > min_conf) & (confidences < max_conf)
    bucket_is_correct = (y_preds == y_true)[bucket_filter]
    bucket_size = bucket_is_correct.shape[0]
    bucket_acc = bucket_is_correct.mean()
    avg_conf = confidences[bucket_filter].mean()

    return bucket_acc, avg_conf, bucket_size


def check_exp(paths):
    ok_list = []
    not_ok_list = []
    eps_ok_list = []
    num_samples_list = []
    for path in paths:
        eps_to_adv_results_dict = get_results(*path.split('/'))
        ok_list.append(path)
        eps_ok = [eps for eps in eps_to_adv_results_dict.keys() if isinstance(eps, int)]
        eps_ok_list.append(eps_ok)
        n_samples = []
        for eps in eps_ok:
            if isinstance(eps, int):
                n_samples.append(eps_to_adv_results_dict[eps]['ground_truth'].shape[0])
        num_samples_list.append(n_samples)
        not_ok_list.extend(eps_to_adv_results_dict['not_ok_eps_path'])

    print("<<< OK EXP :D >>>")
    for ok_exp, eps_ok, n_samples in zip(ok_list, eps_ok_list, num_samples_list):
        print(f"{ok_exp} -> {eps_ok} -> {n_samples}")
    print("<<< NOT OK EXP :( >>>")
    for not_ok in not_ok_list:
        print(not_ok)

    return ok_list


def print_pairwise_distance(eps_to_adv_results_dict):
    clean_probs = eps_to_adv_results_dict[0]['mean_probs']
    adv_probs = eps_to_adv_results_dict[4]['mean_probs']
    pairwise_distance = torch.nn.functional.pairwise_distance(clean_probs, adv_probs)
    print(pairwise_distance.mean(), pairwise_distance.std())
    print("")


def compute_ece(y_true, out_probs, compute_abs=True):
    y_true = y_true.numpy()
    out_probs = out_probs.numpy()

    # num_buckets = 10
    # buckets = np.linspace(0, 1, num=10)

    conf_step = 0.1
    bin_lowers = np.arange(0, 1, step=conf_step)
    bin_uppers = bin_lowers + conf_step
    y_preds = out_probs.argmax(axis=1)
    confidences = out_probs.max(axis=1)
    n_samples = y_true.size

    bucket_accs = []
    bucket_sizes = []
    avg_conf_list = []
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        bucket_filter = (confidences > bin_lower) & (confidences < bin_upper)
        bucket_is_correct = (y_preds == y_true)[bucket_filter]
        bucket_size = bucket_is_correct.shape[0]
        bucket_acc = 0 if np.isnan(bucket_is_correct.mean()) else bucket_is_correct.mean()
        avg_conf = 0 if np.isnan(confidences[bucket_filter].mean()) else confidences[bucket_filter].mean()
        # bin_acc, avg_conf, bin_size = get_bucket_acc(confidences, y_true, y_preds, bin_lower, bin_upper)
        diff_conf = (avg_conf - bucket_acc)
        diff_conf = np.abs(diff_conf) if compute_abs else diff_conf
        ece += diff_conf * (bucket_size/n_samples) if bucket_size > 0 else 0

        bucket_sizes.append(bucket_size)
        bucket_accs.append(bucket_acc)
        avg_conf_list.append(avg_conf)

        avg_acc = (y_preds == y_true).mean()
        avg_conf = confidences.mean()

    return ece, np.array(bucket_accs), np.array(bucket_sizes), np.array(avg_conf_list), bin_lowers, bin_uppers, (avg_acc, avg_conf)


######################################################################################################################


def plot_all_scatters(paths, ds="dsname", metric='entropy_of_mean', figsize=(5, 5), figtitle='all_scatter.pdf'):
    """
    Scatter plot of point before and after the attack, green are correctly classified samples and
    red are not correctly classified.
    Args:
        paths:
        metric:
        figsize:
        figtitle:

    Returns:

    """
    nplots = len(paths)
    ncols = 4 if nplots >= 4 else nplots
    ncols = min(ncols, nplots)
    nrows = nplots // ncols + int(nplots % ncols != 0)
    fig, axs = viz.create_figure(nrows, ncols, figsize=figsize, fontsize=15, squeeze=False)

    if ds not in ["imagenet", "cifar10"]:
        print("plot_all_scatters Dataset not supported")
        return

    save_path = f"figures/MI_scatter/{ds_name}/"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for plot_i, path in enumerate(paths):
        try:
            eps_to_adv_results_dict = get_results(*path.split('/'))
            model_name = path.split('/')[-3]

            eps = 4 if 'imagenet' in path else 8
            i, j = plot_i // ncols, plot_i % ncols

            ax = axs[i, j]

            metric_clean = eps_to_adv_results_dict[0][metric]
            metric_adv = eps_to_adv_results_dict[eps][metric]

            nsamples = min(metric_clean.shape[0], metric_adv.shape[0])
            metric_clean = metric_clean[:nsamples]
            metric_adv = metric_adv[:nsamples]

            # pearson = scipy.stats.pearsonr(metric_clean, metric_adv)
            # spearman = scipy.stats.spearmanr(metric_clean, metric_adv)

            y = eps_to_adv_results_dict[0]['ground_truth']
            y_pred = eps_to_adv_results_dict[0]['preds']
            correct = (y_pred == y).numpy()
            clean_acc = correct.mean()

            title = f"{model_name}\n"
            title += f"Accuracy: {clean_acc}\n"
            # title += f"pearson: {pearson[0]:.2f}, p-val: {pearson[1]:.2f}\n"
            # title += f"spearman: {spearman[0]:.2f}, p-val: {spearman[1]:.2f}"
            ax.set_title(title)


            ax.scatter(metric_clean[correct], metric_adv[correct], alpha=0.1, label='correct', color='green')
            ax.scatter(metric_clean[~correct], metric_adv[~correct], alpha=0.1, label='wrong', color='red')
            ax.set_xlabel('Entropy before attack (H)')

        except Exception as e:
            print(e)
            print("")

    for i in range(nrows):
        axs[i, 0].set_ylabel('Entropy after attack (Hadv)')

    axs[0,0].legend(loc='upper left')

    fig.tight_layout()
    fig.show()
    fig.savefig(f"{save_path}{figtitle}.pdf")

def plot_cal_curves_and_hist(paths, ds="dsname", figsize=(5, 5), figtitle='calibration_curves'):
    """
    Prende le paths degli attacchi Oatk e aggiunge gli Uatk. Vengono generati un reliability ed un histogram
    per ogni modello e salvati.
    """

    if ds not in ["imagenet", "cifar10"]:
        print("plot_cal_curves_and_hist Dataset not supported")
        return

    save_path = f"figures/calibration_hist/{ds}/"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for plot_i, path in enumerate(paths):
        eps = 4 if 'imagenet' in path else 8
        fig, axs = viz.create_figure(2, 1, figsize=figsize, fontsize=15, squeeze=False, rateo=[2, 1])

        path_splitted = path.split('/')
        eps_to_adv_results_dict = get_results(*path_splitted)

        # CLEAN CURVE
        y_true = eps_to_adv_results_dict[0]['ground_truth']
        out_probs = eps_to_adv_results_dict[0]['mean_probs']
        clean_ece, clean_bucket_accs, clean_bucket_sizes, clean_avg_conf_list, bin_lowers, bin_uppers, avg_acc_conf = compute_ece(y_true, out_probs)
        axs[0,0].plot(clean_avg_conf_list, clean_bucket_accs, label=f"clean (ECE={clean_ece:.3f})", marker='o', color=COLORS[0])

        axs[1,0].fill_between(clean_avg_conf_list, 0, clean_bucket_sizes, alpha=0.3, color=COLORS[0])
        axs[1,0].plot(clean_avg_conf_list, clean_bucket_sizes, alpha=0.5, linestyle='dashed', marker='o', color=COLORS[0])

        avg_acc, avg_conf = avg_acc_conf
        axs[1, 0].axvline(x=avg_conf, label="avg confidence", linestyle='dashed', color=COLORS[0])
        axs[1, 0].axvline(x=avg_acc, label="Accuracy", color=COLORS[0])

        # OVER CURVE
        y_true = eps_to_adv_results_dict[eps]['ground_truth']
        out_probs = eps_to_adv_results_dict[eps]['mean_probs']
        rob_acc = (y_true.numpy() == out_probs.numpy().argmax(axis=1)).mean()
        adv_ece, adv_bucket_accs, adv_bucket_sizes, adv_avg_conf_list, bin_lowers, bin_uppers, avg_acc_conf = compute_ece(y_true, out_probs)
        axs[0,0].plot(adv_avg_conf_list, adv_bucket_accs, label=f"adv-O (ECE={adv_ece:.3f})", marker='o', color=COLORS[1])

        # adv_bucket_sizes = adv_bucket_sizes / adv_bucket_sizes.sum()
        axs[1,0].fill_between(adv_avg_conf_list, 0, adv_bucket_sizes, alpha=0.3, color=COLORS[1])
        axs[1,0].plot(adv_avg_conf_list, adv_bucket_sizes, alpha=0.5, linestyle='dashed', marker='o', color=COLORS[1])

        avg_acc, avg_conf = avg_acc_conf
        axs[1, 0].axvline(x=avg_conf, label="avg confidence", linestyle='dashed', color=COLORS[1])
        axs[1, 0].axvline(x=avg_acc, label="Accuracy", color=COLORS[1])


        # UNDER CURVE
        eps_to_adv_results_dict = get_results(*path_splitted[:-2], 'U-atk', 'Shake')
        y_true = eps_to_adv_results_dict[eps]['ground_truth']
        out_probs = eps_to_adv_results_dict[eps]['mean_probs']
        urob_acc = (y_true.numpy() == out_probs.numpy().argmax(axis=1)).mean()
        adv_ece, adv_bucket_accs, adv_bucket_sizes, adv_avg_conf_list, bin_lowers, bin_uppers, avg_acc_conf = compute_ece(y_true, out_probs)
        axs[0,0].plot(adv_avg_conf_list, adv_bucket_accs, label=f"adv-U (ECE={adv_ece:.3f})", marker='o', color=COLORS[2])

        # adv_bucket_sizes = adv_bucket_sizes / adv_bucket_sizes.sum()
        axs[1,0].fill_between(adv_avg_conf_list, 0, adv_bucket_sizes, alpha=0.3, color=COLORS[2])
        axs[1,0].plot(adv_avg_conf_list, adv_bucket_sizes, alpha=0.5, linestyle='dashed', marker='o', color=COLORS[2])

        avg_acc, avg_conf = avg_acc_conf
        axs[1,0].axvline(x=avg_conf, label="avg confidence", linestyle='dashed', color=COLORS[2])
        axs[1,0].axvline(x=avg_acc, label="Accuracy", color=COLORS[2])



        # CUSTOMIZATION

        axs[0,0].set_ylabel('fraction of correct predictions')
        axs[1, 0].set_ylabel('Number of samples')
        axs[0,0].plot([0, 1], [0, 1], linestyle='dashed', color='grey')

        model_name = path_splitted[-3 if 'semi_robust' in path else -4]

        figname=figtitle+f"_{model_name}"
        print(f"Shake {model_name=}, {avg_acc_conf=}")

        title = f"{model_name}\nacc (advO / advU)\n({rob_acc:.3f} / {urob_acc:.3f})"

        fig.suptitle(title)

        axs[0,0].set_ylabel('fraction of correct predictions')
        axs[1, 0].set_ylabel('Number of samples')
        axs[0,0].plot([0, 1], [0, 1], linestyle='dashed', color='grey')

        axs[0,0].set_ylim(0, 1)
        axs[0,0].set_xlim(0, 1)
        axs[1,0].set_xlim(0, 1)

        axs[0,0].grid('on')
        axs[1,0].grid('on')

        axs[0,0].legend(loc='upper left')

        acc_line = Line2D([0], [0], color="black")
        conf_line = Line2D([0], [0], color="black", linestyle="--")

        labels = ["Accuracy", "Avg. confidence"]
        axs[1,0].legend([acc_line, conf_line], labels, loc='upper left')

        axs[0, 0].set_xlabel('predicted probability')
        axs[1, 0].set_xlabel('predicted probability')

        fig.tight_layout()
        fig.show()
        fig.savefig(f"{save_path}{figname}.pdf")

    print("")


def plot_entropy_gap(paths, ds="cifar"):
    """
    Plot models entropy for the different types of attacks.
    On x axis there is model alias (M*), y axis mean entropy.
    """

    if ds not in ["imagenet", "cifar"]:
        print("plot_entropy_gap Dataset not supported")
        return

    save_path = "figures/entropy_gap/"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    if ds == "imagenet":
        paths.pop(0)

    eps = 8 if ds == 'cifar' else 4

    model_dict = {}
    model_name_list = []

    for i, model in enumerate(paths[0]):
        path_splitted = model.split('/')
        model_name = path_splitted[path_splitted.index('None') + (1 if not 'naive' in model else -1)]
        model_dict[model_name] = {"Clean": 0, "Stab": 0, "Shake": 0, "AutoTarget": 0}
        # model_name_list.append(model_name)
        model_name_list.append(f"M{i+1}")

    print(model_name_list)

    for attack_path in paths[::2]:
        for model in attack_path:
            path_splitted = model.split('/')
            model_name = path_splitted[path_splitted.index('None') + (1 if not 'naive' in model else -1)]
            attack_type = path_splitted[-1]

            res = get_results(*path_splitted)

            model_dict[model_name]["Clean"] = res[0]['entropy_of_mean'].mean().item()
            model_dict[model_name][attack_type] = res[eps]['entropy_of_mean'].mean().item()

    fig, ax = viz.create_figure(figsize=(15, 15))

    for idx, (key, item) in enumerate(model_dict.items()):
        print(key, item)
        ax.plot(idx, item["Clean"], color=COLORS[0], marker=ALL_MARKERS[0], markersize=10)
        ax.plot(idx, item["Stab"], color=COLORS[1], marker=ALL_MARKERS[1], markersize=10)
        ax.plot(idx, item["Shake"], color=COLORS[2], marker=ALL_MARKERS[2], markersize=10)
        ax.plot(idx, item["AutoTarget"], color=COLORS[3], marker=ALL_MARKERS[3], markersize=10)

    labels = ["Clean", "Stab", "Shake", "AutoTarget"]
    markers = [Line2D([], [], color=COLORS[i], marker=ALL_MARKERS[i], linestyle='None',
                      markersize=15, label=key) for i, key in enumerate(labels)]

    ax.legend(markers, labels, loc='upper right')
    title = f"ENTROPY_GAP_{ds}"
    ax.set_title(title)

    ax.set_xticks(range(len(model_name_list)))
    ax.set_xticklabels(model_name_list, rotation=45, ha="right")

    ax.set_xlabel('Models')
    ax.set_ylabel('H(x)')
    ax.grid("on")
    # fig.subplots_adjust(bottom=0.30)
    fig.show()
    fig.savefig(f"{save_path}{title}.pdf")




if __name__ == '__main__':
    atk_type_and_name_list = ['O-atk/Stab', "O-atk/AutoTarget",  'U-atk/Shake']
    datasets = ['cifar10', 'imagenet']
    robusts = [True]

    dict_terms = {'semi_robust': 'robust',
                  'naive_robust': 'naive'}

    # PATHS STRUCTURE: 0-C10 stab, 1-IMG stab, 2-C10 AT, 3-IMG AT, 4-C10 shake, 5-IMG shake
    paths = get_paths(datasets, atk_type_and_name_list, robusts)

    # ------------------------ CALIBRATOIN CURVE
    # PRENDO SOLO STAB
    for path_list in paths[:2]:
        robustness_level = dict_terms[path_list[0].split('/')[2]]
        ds_name = path_list[0].split('/')[3]
        attack_type = path_list[0].split('/')[-1]

        title = f"calibration_curve_{robustness_level}_{ds_name}"
        # plot_cal_curves_and_hist(path_list, ds=ds_name, figsize=(5,10), figtitle=title)

    # ------------------------ MI SAMPLES PLOTS
    for path_list in paths:
        robustness_level = dict_terms[path_list[0].split('/')[2]]
        ds_name = path_list[0].split('/')[3]
        attack_type = path_list[0].split('/')[-1]
        title = f"scatter_{robustness_level}_{ds_name}_{attack_type}"
        # plot_all_scatters(path_list, ds=ds_name, figtitle=title)

    # ----------------------- ENTROPY GAP PLOTS
    # plot_entropy_gap(paths, ds="cifar")
    # plot_entropy_gap(paths, ds="imagenet")









