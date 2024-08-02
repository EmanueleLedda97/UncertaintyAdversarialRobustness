
from utils.utils import join, my_load
import os
import numpy as np
import torch

import utils.visualisation as viz

from matplotlib.colors import TABLEAU_COLORS
import utils.utils as utils
import warnings

from utils.constants import CIFAR10_ROBUST_MODELS, IMAGENET_ROBUST_MODELS, \
    CIFAR10_NAIVE_MODELS, IMAGENET_NAIVE_MODELS, \
    cifar10_model_dict, imagenet_model_dict
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


COLORS = ['tab:green', 'tab:orange', 'tab:blue', 'tab:red']
ALL_COLORS = list(TABLEAU_COLORS.keys())
ALL_MARKERS = list("o*sP^<>1234vXD")

warnings.simplefilter("ignore")

"""
PLOT FILE WITH ONLY ESSENTIAL PLOTS
"""



def get_paths(datasets, atk_type_and_name_list, robusts=[True], verbose=True):
    path_ok_list = []
    for atk_type_and_name in atk_type_and_name_list:
        if verbose: print('#' * 50)
        atk_type = f"Attack type: {atk_type_and_name}"
        if verbose: print(f"#{atk_type.center(48, ' ')}#")
        if verbose: print('#' * 50)
        for dataset in datasets:
            s = f" Dataset: {dataset} "
            if verbose: print(s.center(50, '+'))
            for robust in robusts:
                s = " Robust Models " if robust else " Naive Models "
                if verbose: print(s.center(50, '-'))
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

                path_ok_list.append(check_exp(paths, verbose))
                # try:
                #     details = f"_{dataset}_{atk_type_and_name.split('/')[1]}_{s.split()[0]}"
                #     # plot_conf_displacement(path_ok_list)
                #     plot_cal_curves(path_ok_list, figname=f"calibration_curve_{details}")
                # except:
                #     print("")
                # plot_ECE(path_ok_list, figname=f"ECE_{details}")
                if verbose: print("")
            if verbose: print('/' * 50)
        if verbose: print("\\" * 50)
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


def check_exp(paths, verbose=True):
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

    if verbose: print("<<< OK EXP :D >>>")
    for ok_exp, eps_ok, n_samples in zip(ok_list, eps_ok_list, num_samples_list):
        if verbose: print(f"{ok_exp} -> {eps_ok} -> {n_samples}")
    if verbose: print("<<< NOT OK EXP :( >>>")
    for not_ok in not_ok_list:
        if verbose: print(not_ok)

    return ok_list


def print_pairwise_distance(eps_to_adv_results_dict):
    clean_probs = eps_to_adv_results_dict[0]['mean_probs']
    adv_probs = eps_to_adv_results_dict[4]['mean_probs']
    pairwise_distance = torch.nn.functional.pairwise_distance(clean_probs, adv_probs)
    print(pairwise_distance.mean(), pairwise_distance.std())
    print("")


def compute_ece(y_true, out_probs, k, compute_abs=True):
    y_true = y_true.numpy()
    out_probs = out_probs.numpy()

    # num_buckets = 10
    # buckets = np.linspace(0, 1, num=10)

    conf_step = 1/k
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

        if bucket_size != 0:
            bucket_sizes.append(bucket_size)
            bucket_accs.append(bucket_acc)
            avg_conf_list.append(avg_conf)

        # else:
        #     bucket_sizes.append(0)
        #     bucket_accs.append(bucket_acc)
        #     avg_conf_list.append(avg_conf)


    avg_acc = (y_preds == y_true).mean()
    avg_conf = confidences.mean()

    return ece, np.array(bucket_accs), np.array(bucket_sizes), np.array(avg_conf_list), bin_lowers, bin_uppers, (avg_acc, avg_conf)


def fix_values(path, metrics, atk_to_fix):
    """
    Fix the values of the given metrics by keeping the best value, wrt H(x)
    Args:
        path: path to the file
        metrics: metrics to fix
        atk_to_fix: attack to fix: Stab, Shake

    Returns: The metrics that we wanted after updating the data

    """

    if atk_to_fix not in ["Stab", "Shake"]:
        assert f"{atk_to_fix} is not a valit attack type"

    eps = 4 if 'imagenet' in path else 8
    path_splitted = path.split('/')

    # GET CLEAN
    clean_results_dict = get_results(*path_splitted)[0]
    clean_entropies = clean_results_dict['entropy_of_mean']

    if metrics not in clean_results_dict.keys():
        assert f"{metrics} not a valid metrics, choose from {clean_results_dict.keys()}"

    mask = None
    adv_results_dict = None

    # GET ATKMETRICS
    if atk_to_fix == "Stab":
        adv_results_dict = get_results(*path_splitted[:-2], 'O-atk', 'Stab')[eps]
        atk_entropies = adv_results_dict['entropy_of_mean']
        mask = clean_entropies <= atk_entropies

    if atk_to_fix == "Shake":
        adv_results_dict = get_results(*path_splitted[:-2], 'U-atk', 'Shake')[eps]
        atk_entropies = adv_results_dict['entropy_of_mean']
        mask = clean_entropies >= atk_entropies

    adv_results_dict[metrics][mask] = clean_results_dict[metrics][mask]

    return adv_results_dict[metrics]

######################################################################################################################

def plot_ece_and_hist(y_true, out_probs, ax_up, label, color, dict_plot_params, k=15):
    ece, bucket_accs, bucket_sizes, avg_conf_list, bin_lowers, bin_uppers, avg_acc_conf = compute_ece(
        y_true, out_probs, k)

    # normalization = (bucket_sizes <= k) * np.arange(1, len(bucket_sizes) + 1, 1) / len(bucket_sizes)
    # avg_conf_list[bucket_sizes <= k] = normalization[bucket_sizes <= k]
    # bucket_accs[bucket_sizes <= k] = normalization[bucket_sizes <= k]
    #
    # area_x = np.concatenate(([0.0], avg_conf_list, [1.0]))
    # area_y = np.concatenate(([0.0], bucket_accs, [1.0]))
    #
    # area = trapz(area_y, area_x)
    area, bucket_accs, bucket_sizes, avg_conf_list, bin_lowers, bin_uppers, avg_acc_conf = compute_ece(
        y_true, out_probs, k, compute_abs=False)

    # print(f"{color=} -- {area=}")

    ax_up.plot(avg_conf_list, bucket_accs, label=f"label {ece:.3f}", marker='o',
               color=color, **dict_plot_params)
    # Histogram
    ax_up.fill_between(avg_conf_list, 0, bucket_sizes / bucket_sizes.sum(), alpha=0.3, color=color)
    ax_up.plot(avg_conf_list, bucket_sizes / bucket_sizes.sum(), alpha=0.5, linestyle='dashed', marker='d',
               color=color, **dict_plot_params)

    acc, avg_conf = avg_acc_conf

    # Vertical line
    # ax_up.axvline(x=avg_conf, label="avg confidence", linestyle='dashed', color=color, **dict_plot_params)
    # ax_up.axvline(x=acc, label="Accuracy", color=color, **dict_plot_params)

    return ece, acc, area


def plot_ece_hist_original(y_true, out_probs, ax_down, ax_up, label, color, dict_plot_params):
    ece, bucket_accs, bucket_sizes, avg_conf_list, bin_lowers, bin_uppers, avg_acc_conf = compute_ece(
        y_true, out_probs)

    ax_up.plot(avg_conf_list, bucket_accs, label=f"label {ece:.3f}", marker='o',
               color=color, **dict_plot_params)
    # Histogram
    ax_down.fill_between(avg_conf_list, 0, bucket_sizes, alpha=0.3, color=color)
    ax_down.plot(avg_conf_list, bucket_sizes, alpha=0.5, linestyle='dashed', marker='o',
                 color=color, **dict_plot_params)

    acc, avg_conf = avg_acc_conf

    # Vertical line
    ax_down.axvline(x=avg_conf, label="avg confidence", lw=10, linestyle='dashed', color=color)
    ax_down.axvline(x=acc, label="Accuracy", lw=10, color=color)

    return ece, acc


# -----------------------------------------------------------------------

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

            attack_type = path.split('/')[-1]

            eps = 4 if 'imagenet' in path else 8
            i, j = plot_i // ncols, plot_i % ncols

            ax = axs[i, j]

            metric_clean = eps_to_adv_results_dict[0][metric]
            metric_adv = fix_values(path, metric, attack_type)

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
    fig.savefig(f"{save_path}{figtitle}.pdf", bbox_inches='tight')


def plot_cal_curves_and_hist_single(paths, ds="dsname", figsize=(15, 10), figtitle='calibration_curves'):
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

    dict_plot_params = {"lw": 15, "ms": 50}

    for plot_i, path in enumerate(paths):
        eps = 4 if 'imagenet' in path else 8

        figsize = (30,30)

        fig, axs = viz.create_figure(1, 1, figsize=figsize, squeeze=True, fontsize=100)

        ax_up = axs

        path_splitted = path.split('/')
        eps_to_adv_results_dict = get_results(*path_splitted)

        # CLEAN
        y_true = eps_to_adv_results_dict[0]['ground_truth']
        out_probs = eps_to_adv_results_dict[0]['mean_probs']
        clean_ece, clean_acc = plot_ece_and_hist(y_true, out_probs, ax_up, "clean ECE", COLORS[0], dict_plot_params)

        # OVER CURVE
        y_true = fix_values(path, 'ground_truth', 'Stab')
        out_probs = fix_values(path, 'mean_probs', 'Stab')
        over_ece, over_acc = plot_ece_and_hist(y_true, out_probs, ax_up, "Adv-O", COLORS[1], dict_plot_params)

        # UNDER
        y_true = fix_values(path, 'ground_truth', 'Shake')
        out_probs = fix_values(path, 'mean_probs', 'Shake')
        under_ece, under_acc = plot_ece_and_hist(y_true, out_probs, ax_up, "Adv-U", COLORS[2], dict_plot_params)

        # CUSTOMIZATION
        ax_up.set_ylabel('fraction of correct predictions')
        #ax_up.plot([0, 1], [0, 1], linestyle='dashed', color='grey')

        model_name = path_splitted[-3 if 'semi_robust' in path else -4]

        figname=figtitle+f"_{model_name}"

        print(f"{model_name=} {clean_ece=} {over_ece=} {under_ece=}")

        title = f"{model_name}"#\nacc (advO / advU)\n({over_acc:.3f} / {under_acc:.3f})"
        title = f"I{plot_i+1}" if 'imagenet' in path else f"M{plot_i+1}"

        ax_up.set_title(title)

        ttl = ax_up.title
        ttl.set_position([.5, 1.05])

        ax_up.set_ylabel('% of correct predictions', labelpad=15)
        ax_up.set_xlabel('predicted probability')

        ax_up.plot([0, 1], [0, 1], linestyle='dashed', color='grey')

        ax_up.set_ylim(0.001, 1)
        ax_up.set_yticks([0.25, 0.5, 0.75,1], minor=False)
        ax_up.set_yticklabels(["0.25", "0.5", "0.75","1"], fontdict=None, minor=False)

        ax_up.set_xlim(0, 1)
        ax_up.set_xticks([0, 0.25, 0.5, 0.75, 1], minor=False)
        ax_up.set_xticklabels(['0', "0.25", "0.5", "0.75", "1"], fontdict=None, minor=False)

        ax_up.grid('on')

        fig.tight_layout()
        fig.show()
        fig.savefig(f"{save_path}{figname}.pdf", dpi=400)


def plot_cal_curves_and_hist_allinone(paths, ncols=5, curvedim=4, figname='calibration_curves'):
    """
    Plot calibration and histogram all in one single plot.
    Also save ECE scores in the same folder.
    """

    save_path = "figures/calibration_hist/"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    nplots = len(paths)
    ncols = 5 if nplots == 5 else ncols
    ncols = min(ncols, nplots)
    nrows = nplots // ncols + int(nplots % ncols != 0)

    scale = 20

    figsize = (nrows * scale, ncols * scale)

    fig, axs = viz.create_figure(nrows, ncols, figsize=figsize, fontsize=120, squeeze=True,
                                 gridspec_kw = {'wspace':0.1, 'hspace':0.2})  # hspace gestisce spazio tra plot

    models_ece = {}

    dict_plot_params = {"lw":15, "ms":50}

    for plot_i, path in enumerate(paths):
        path_splitted = path.split('/')

        model_name = path_splitted[-3 if 'semi_robust' in path else -4]
        models_ece[model_name] = {}

        eps = 4 if 'imagenet' in path else 8

        i, j = (plot_i // ncols), plot_i % ncols

        ax_up = axs[i, j]  # asse delle curve

        eps_to_adv_results_dict = get_results(*path_splitted)

        # CLEAN
        y_true = eps_to_adv_results_dict[0]['ground_truth']
        out_probs = eps_to_adv_results_dict[0]['mean_probs']

        clean_ece, clean_acc, clean_area = plot_ece_and_hist(y_true, out_probs, ax_up, "clean ECE", COLORS[0],
                                                             dict_plot_params)

        # OVER CURVE
        y_true = fix_values(path, 'ground_truth', 'Stab')
        out_probs = fix_values(path, 'mean_probs', 'Stab')

        over_ece, over_acc, over_area = plot_ece_and_hist(y_true, out_probs, ax_up, "Adv-O", COLORS[1],
                                                          dict_plot_params)

        # UNDER
        y_true = fix_values(path, 'ground_truth', 'Shake')
        out_probs = fix_values(path, 'mean_probs', 'Shake')

        under_ece, under_acc, under_area = plot_ece_and_hist(y_true, out_probs, ax_up, "Adv-U", COLORS[2],
                                                             dict_plot_params)

        models_ece[model_name] = {"clean_ece": clean_ece, "over_ece": over_ece, "under_ece": under_ece,
                                  "clean_acc": clean_acc, "over_acc": over_acc, "under_acc": under_acc,
                                  "clean_area": clean_area, "over_area": over_area, "under_area": under_area}
        # CUSTOMIZATION
        # print(f"{models_ece[model_name]}")

        # CURVE TITLE
        model_title = f"{model_name}" #\nacc (clean / advO / advU)\n({clean_acc:.3f} / {over_acc:.3f} / {under_acc:.3f})"
        model_title = f"I{plot_i+1}" if 'imagenet' in path else f"C{plot_i+1}"
        ax_up.set_title(model_title)

        ax_up.plot([0, 1], [0, 1], linestyle='dashed', color='grey', **dict_plot_params)

        ax_up.grid('on', alpha=0.5)

        # SET LABELS
        ax_up.set_ylim(0, 1)
        ax_up.set_yticks([0, 0.25, 0.5, 0.75, 1], minor=False)
        ax_up.set_yticklabels(['0', "0.25", "0.5", "0.75", "1"], fontdict=None, minor=False)
        ax_up.set_xlim(0, 1)
        ax_up.set_xticks([0, 0.25, 0.5, 0.75, 1], minor=False)
        ax_up.set_xticklabels(['0', "0.25", "0.5", "0.75", "1"], fontdict=None, minor=False)

        # HISTOGRAM
        if j != 0:  # Remove tick for internal plots
            ax_up.set_yticklabels([])
        if i!= nrows-1:
            ax_up.set_xticklabels([])

    legend_lines = [ Line2D([0], [0], color=COLORS[i], **dict_plot_params) for i in range(3)]
    labels = ["Clean", "Oatk", "Uatk"]

    anchor = (0.5, 1.02) if 'imagenet' in path else (0.5, 0.95)
    fig.legend(legend_lines, labels, loc='upper center', ncols=3, frameon=False, bbox_to_anchor=anchor)

    axs[-1,0].set_ylim(0.00001, 1)
    for col in range(ncols):
        axs[-1, col].set_xlabel('predicted probability')
        axs[-1, col].set_xlim(0, 1)

    for row in range(0, nrows):
        axs[row, 0].set_ylabel('% of correct\npredictions', labelpad=15)

    # Adjust x label for remove plots
    if nplots < ncols*nrows:
        plot_i = nplots
        while plot_i < ncols*nrows:
            i, j = (plot_i // ncols), plot_i % ncols
            fig.delaxes(axs[i,j])
            plot_i+=1
            axs[i-1, j].set_xlabel('predicted probability')
            axs[i-1, j].set_xlim(0, 1)
            axs[i-1, j].set_xticks([0, 0.25, 0.5, 0.75, 1], minor=False)
            axs[i-1, j].set_xticklabels(['0', "0.25", "0.5", "0.75", "1"], fontdict=None, minor=False)


    fig.tight_layout()
    fig.savefig(f"{save_path}{figname}.pdf", bbox_inches='tight')
    utils.my_save(models_ece, f"{save_path}{figname}.pkl")


def plot_entropy_gap_points(paths, ds="cifar"):
    """
    Plot scatterplot of model's entropy for different types of attacks.
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
            model_dict[model_name][attack_type] = fix_values(path, "entropy_of_mean", attack_type).mean().item()

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
    title = f"ENTROPY_GAP_POINTS_{ds}"
    ax.set_title(title)

    ax.set_xticks(range(len(model_name_list)))
    ax.set_xticklabels(model_name_list, rotation=45, ha="right")

    ax.set_xlabel('Models')
    ax.set_ylabel('H(x)')
    ax.grid("on")
    # fig.subplots_adjust(bottom=0.30)
    fig.show()
    fig.savefig(f"{save_path}{title}.pdf", bbox_inches='tight')


def plot_entropy_gap_bars(paths, ds="cifar"):
    """
    Plot bar plot of model's entropy for different types of attacks.
    On x axis there is RB model name, y axis mean entropy.
    """

    save_path = "figures/entropy_gap/"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    if ds == "imagenet":
        paths.pop(0)

    eps = 8 if ds == 'cifar' else 4

    model_dict = {}
    model_name_list = []

    for model in paths[0]:
        path_splitted = model.split('/')
        model_name = path_splitted[path_splitted.index('None') + (1 if not 'naive' in model else -1)]
        model_dict[model_name] = {"Clean": 0, "Stab": 0, "Shake": 0, "AutoTarget": 0}
        model_name_list.append(model_name)

    print(model_name_list)

    for attack_path in paths[::2]:
        for model in attack_path:
            path_splitted = model.split('/')
            model_name = path_splitted[path_splitted.index('None') + (1 if not 'naive' in model else -1)]
            attack_type = path_splitted[-1]

            res = get_results(*path_splitted)


            model_dict[model_name]["Clean"] = res[0]['entropy_of_mean'].mean().item()
            model_dict[model_name][attack_type] = fix_values(path, "entropy_of_mean", attack_type).mean().item()

    fig, ax = viz.create_figure(figsize=(15, 15))

    width = 0.25  # the width of the bars
    multiplier = 0
    x_ticks = np.arange(len(model_name_list)) * 2

    for idx, (key, item) in enumerate(model_dict.items()):
        offset = width * multiplier

        print(key, item)
        ax.bar(x_ticks[idx], item["Clean"], width, color=COLORS[0], label="Clean")
        ax.bar(x_ticks[idx] + (width * 1), item["Stab"], width, color=COLORS[1], label="Stab")
        ax.bar(x_ticks[idx] + (width * 2), item["Shake"], width, color=COLORS[2], label="Shake")
        ax.bar(x_ticks[idx] + (width * 3), item["AutoTarget"], width, color=COLORS[3], label="AutoTarget")

    labels = ["Clean", "Stab", "Shake", "AutoTarget"]
    markers = [mpatches.Patch(color=COLORS[i], label=key) for i, key in enumerate(labels)]

    ax.legend(markers, labels, loc='upper right')

    title = f"ENTROPY_GAP_BARS_{ds}"
    ax.set_title(title)

    ax.set_xticks(x_ticks + width * 2)
    ax.set_xticklabels(model_name_list, rotation=45, ha="right")

    ax.set_xlabel('Models')
    ax.set_ylabel('H(x)')
    fig.show()
    fig.savefig(f"{save_path}{title}.pdf", bbox_inches='tight')


def plot_violin_single(paths, ds, figsize=(7, 7), figtitle='violin_plot'):
    """
    Plot calibration and histogram all in one single plot.
    """

    save_path = f"figures/violin_plots/{ds}/"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for plot_i, path in enumerate(paths):

        fig, axs = viz.create_figure(1, 1, figsize=figsize, fontsize=20, squeeze=True)

        plot_i = [1]

        eps = 4 if 'imagenet' in path else 8

        path_splitted = path.split('/')
        eps_to_adv_results_dict = get_results(*path_splitted)

        clean_entropies = eps_to_adv_results_dict[0]['entropy_of_mean']

        oatk_entropies = fix_values(path,'entropy_of_mean', 'Stab')

        uatk_entropies = fix_values(path,'entropy_of_mean', 'Shake')

        # print(f"{clean_entropies.shape=}, {oatk_entropies.shape=}, {uatk_entropies.shape=}")

        # --------------------------------- CLEAN VIOLIN
        v1 = axs.violinplot(clean_entropies, points=100, positions=plot_i,
                            showmeans=True, showextrema=False, showmedians=False)

        v1["cmeans"].set_edgecolor(COLORS[0])
        v1["cmeans"].set_linewidth(2)
        segment = v1["cmeans"].get_segments()
        segment[0][1][0] = 1
        v1["cmeans"].set_segments(segment)

        for b in v1['bodies']:
            # get the center
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            # modify the paths to not go further right than the center
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
            b.set_color(COLORS[0])

        # ----------------------------------- OVER VIOLIN
        v2 = axs.violinplot(oatk_entropies, points=100, positions=plot_i,
                            showmeans=True, showextrema=False, showmedians=False)

        v2["cmeans"].set_edgecolor(COLORS[1])
        v2["cmeans"].set_linewidth(2)
        segment = v2["cmeans"].get_segments()
        segment[0][0][0] = 1
        v2["cmeans"].set_segments(segment)

        for b in v2['bodies']:
            # get the center
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            # modify the paths to not go further left than the center
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
            b.set_color(COLORS[1])

        # ------------------------------- UNDER VIOLIN
        v3 = axs.violinplot(uatk_entropies, points=100, positions=plot_i,
                            showmeans=True, showextrema=False, showmedians=False)

        v3["cmeans"].set_edgecolor(COLORS[2])
        v3["cmeans"].set_linewidth(2)
        segment = v3["cmeans"].get_segments()
        segment[0][0][0] = 1
        v3["cmeans"].set_segments(segment)

        for b in v3['bodies']:
            # get the center
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            # modify the paths to not go further left than the center
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
            b.set_color(COLORS[2])

        # axs.set_ylim([-1, 3])

        model_name = path_splitted[-3 if 'semi_robust' in path else -4]

        figname = figtitle + f"_{model_name}"

        axs.set_ylabel("H(x)")
        axs.set_xticks([])

        fig.suptitle(model_name)
        fig.tight_layout()
        fig.show()

        fig.savefig(f"{save_path}{figname}.pdf", bbox_inches='tight')


def plot_violin_all(paths, figsize=(7, 16), figname='violin_plot'):
    """
    Plot calibration and histogram all in one single plot.
    """

    save_path = "figures/violin_plots/"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    fig, axs = viz.create_figure(1, 1, figsize=figsize, squeeze=True, fontsize=30)

    model_name_list = []

    models_avg_entropy = {}

    extend = 0.2 # Di quanto allungare il segmento

    linewidth = 4


    for plot_i, path in enumerate(paths):

        plot_i = [plot_i]

        eps = 4 if 'imagenet' in path else 8

        path_splitted = path.split('/')
        eps_to_adv_results_dict = get_results(*path_splitted)

        model_name = path_splitted[path_splitted.index('None') + (1 if not 'naive' in path else -1)]
        model_name_list.append(f"I{plot_i[0]+1}" if 'imagenet' in path else f"C{plot_i[0]+1}")


        clean_entropies = eps_to_adv_results_dict[0]['entropy_of_mean']

        oatk_entropies = fix_values(path,'entropy_of_mean', 'Stab')

        uatk_entropies = fix_values(path,'entropy_of_mean', 'Shake')

        models_avg_entropy[model_name] = {"clean":clean_entropies.mean().item(),
                                          "oatk":oatk_entropies.mean().item(),
                                          "uatk":uatk_entropies.mean().item()}



        # --------------------------------- CLEAN VIOLIN
        v1 = axs.violinplot(clean_entropies, points=100, positions=plot_i,
                            showmeans=True, showextrema=False, showmedians=False)

        v1["cmeans"].set_edgecolor(COLORS[0])
        v1["cmeans"].set_linewidth(2)
        segment = v1["cmeans"].get_segments()
        segment[0][1][0] = plot_i[0]
        segment[0][0][0] = segment[0][0][0] - extend
        v1["cmeans"].set_segments(segment)
        v1["cmeans"].set_linewidth(linewidth)

        for b in v1['bodies']:
            # get the center
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            # modify the paths to not go further right than the center
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
            b.set_color(COLORS[0])

        # ----------------------------------- OVER VIOLIN
        v2 = axs.violinplot(oatk_entropies, points=100, positions=plot_i,
                            showmeans=True, showextrema=False, showmedians=False)

        v2["cmeans"].set_edgecolor(COLORS[1])
        v2["cmeans"].set_linewidth(2)
        segment = v2["cmeans"].get_segments()
        segment[0][0][0] = plot_i[0]
        segment[0][1][0] = segment[0][1][0] + extend

        v2["cmeans"].set_segments(segment)
        v2["cmeans"].set_linewidth(linewidth)

        for b in v2['bodies']:
            # get the center
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            # modify the paths to not go further left than the center
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
            b.set_color(COLORS[1])

        # ------------------------------- UNDER VIOLIN
        v3 = axs.violinplot(uatk_entropies, points=100, positions=plot_i,
                            showmeans=True, showextrema=False, showmedians=False)

        v3["cmeans"].set_edgecolor(COLORS[2])
        v3["cmeans"].set_linewidth(2)
        segment = v3["cmeans"].get_segments()
        segment[0][0][0] = plot_i[0]
        segment[0][1][0] = segment[0][1][0] + extend
        v3["cmeans"].set_segments(segment)
        v3["cmeans"].set_linewidth(linewidth)


        for b in v3['bodies']:
            # get the center
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            # modify the paths to not go further left than the center
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
            b.set_color(COLORS[2])

    # Y-AXIS
    axs.set_ylabel("H(x)")
    ytick = np.around(np.linspace(0, axs.get_ylim()[1], 7), 1)
    axs.set_yticks(ytick, minor=False)
    axs.set_yticklabels([str(y) for y in ytick], fontdict=None, minor=False)

    # X-AXIS
    axs.set_xlabel("Models")
    axs.set_xticks(np.arange(len(model_name_list)))
    axs.set_xticklabels(model_name_list, rotation=45, ha="right")

    # LEGEND
    legend_lines = [ Line2D([0], [0], color=COLORS[i], lw=linewidth) for i in range(3)]
    labels = ["Clean", "Oatk", "Uatk"]

    fig.legend(legend_lines, labels, loc='upper center', ncols=3, frameon=False, bbox_to_anchor=(0.5, 1.05))

    axs.grid('on')
    # fig.suptitle(figname)
    fig.tight_layout()
    fig.savefig(f"{save_path}{figname}.pdf", bbox_inches='tight')

    utils.my_save(models_avg_entropy, f"{save_path}{figname}.pkl")



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == '__main__':
    atk_type_and_name_list = ['O-atk/Stab', "O-atk/AutoTarget",  'U-atk/Shake']
    datasets = ["cifar10","imagenet"]
    robusts = [True]

    dict_terms = {'semi_robust': 'robust',
                  'naive_robust': 'naive'}

    # PATHS STRUCTURE: 0-C10 stab, 1-IMG stab, 2-C10 AT, 3-IMG AT, 4-C10 shake, 5-IMG shake
    paths = get_paths(datasets, atk_type_and_name_list, robusts, verbose=False)

    # ------------------------ CALIBRATOIN CURVE
    for path_list in paths[:2]:
        robustness_level = dict_terms[path_list[0].split('/')[2]]
        ds_name = path_list[0].split('/')[3]
        title = f"calibration_curve_{robustness_level}_{ds_name}"
        #plot_cal_curves_and_hist_single(path_list, ds=ds_name, figtitle=title)
        plot_cal_curves_and_hist_allinone(path_list, ncols=4, curvedim=5, figname=title)

    # ----------------------- VIOLIN PLOTS
    for path_list in paths[:2]:
        robustness_level = dict_terms[path_list[0].split('/')[2]]
        ds_name = path_list[0].split('/')[3]
        title = f"violinplots_{robustness_level}_{ds_name}"
        #plot_violin_single(path_list, ds=ds_name, figtitle=title)
        plot_violin_all(path_list, figname=title)


    # ------------------------ MI SAMPLES PLOTS
    for path_list in paths:
        robustness_level = dict_terms[path_list[0].split('/')[2]]
        ds_name = path_list[0].split('/')[3]
        attack_type = path_list[0].split('/')[-1]
        title = f"scatter_{robustness_level}_{ds_name}_{attack_type}"
        #plot_all_scatters(path_list, ds=ds_name, figtitle=title)

    # ----------------------- ENTROPY GAP PLOTS
    # plot_entropy_gap_points(paths, ds="cifar")
    # plot_entropy_gap_points(paths, ds="imagenet")
    # plot_entropy_gap_bars(paths, ds="cifar")
    # plot_entropy_gap_bars(paths, ds="imagenet")









