
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
ALL_MARKERS = list("ov^<>1234sP*XD")


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


def plot_all_scatters(paths, metric='entropy_of_mean', figpath='figures/all_scatter.pdf'):
    fig, axs = viz.create_figure(nrows=1, ncols=len(paths), figsize=(7, 7),
                                 squeeze=False, fontsize=30)

    # robust_method = 'Naive' if robustness_level == 'naive_robust' else robust_method

    for i, path in enumerate(paths):
        try:
            eps_to_adv_results_dict = get_results(*path.split('/'))
            model_name = path.split('/')[-3]

            eps_list = list(eps_to_adv_results_dict.keys())
            eps = 4

            metric_clean = eps_to_adv_results_dict[0][metric]
            metric_adv = eps_to_adv_results_dict[eps][metric]

            nsamples = min(metric_clean.shape[0], metric_adv.shape[0])
            metric_clean = metric_clean[:nsamples]
            metric_adv = metric_adv[:nsamples]

            pearson = scipy.stats.pearsonr(metric_clean, metric_adv)
            spearman = scipy.stats.spearmanr(metric_clean, metric_adv)

            y = eps_to_adv_results_dict[0]['ground_truth']
            y_pred = eps_to_adv_results_dict[0]['preds']
            correct = (y_pred == y).numpy()
            clean_acc = correct.mean()

            title = f"{model_name}\n"
            title += f"Accuracy: {clean_acc}\n"
            title += f"pearson: {pearson[0]:.2f}, p-val: {pearson[1]:.2f}\n"
            title += f"spearman: {spearman[0]:.2f}, p-val: {spearman[1]:.2f}"
            axs[0, i].set_title(title)

            # ax = axs[0, i]
            # bins = 10
            # alpha = 0.5
            # ax.hist(metric_clean, bins=bins, alpha=alpha, label='before atk', density=True)
            # ax.hist(metric_adv, bins=bins, alpha=alpha, label='after atk', density=True)
            # # args = {'fill': True, }
            # # sns.kdeplot(metric_clean, alpha=alpha, label='clean')
            # # sns.kdeplot(metric_adv, alpha=alpha, label='adv')
            # ax.set_xlabel('Entropy')
            # ax.set_ylabel('Count')
            # ax.legend()

            ax = axs[0, i]

            ax.scatter(metric_clean[correct], metric_adv[correct], alpha=0.1, label='correct', color='green')
            ax.scatter(metric_clean[~correct], metric_adv[~correct], alpha=0.1, label='wrong', color='red')
            ax.set_xlabel('Entropy before attack (H)')
            # ax.set_ylabel('Entropy after attack (Hadv')
        except Exception as e:
            print(e)
            print("")
    axs[0, 0].set_ylabel('Entropy after attack (Hadv)')

    # title = join(dataset, backbone, robust_method)
    # fig.suptitle(title)
    fig.tight_layout()
    fig.show()
    fig.savefig(figpath)


def plot_correlations(paths):

    fig, axs = viz.create_figure(ncols=1, figsize=(10, 10))

    model_names = []
    pearson = []
    pvalues = []
    spearman = []

    for path in paths:
        path_splitted = path.split('/')
        model_names.append(path_splitted[-3])
        eps_to_adv_results_dict = get_results(*path_splitted)

        metric = 'entropy_of_mean'

        eps = 4
        metric_clean = eps_to_adv_results_dict[0][metric]
        metric_adv = eps_to_adv_results_dict[eps][metric]

        res = scipy.stats.pearsonr(metric_clean, metric_adv)
        pearson.append(res[0])
        pvalues.append(res[1])
        spearman.append(scipy.stats.spearmanr(metric_clean, metric_adv)[0])


    width = 0.3
    multiplier = 0
    measures_list = [pearson, spearman]
    measure_names = ['Pearson', 'Spearman']
    x = np.arange(len(model_names))

    for i, (measure, measure_name) in enumerate(zip(measures_list, measure_names)):
        # if i>0:
        #     break
        offset = width * multiplier
        rects = axs.bar(x + offset, measure, width, label=measure_name)
        # axs.bar_label(rects, padding=3)
        multiplier += 1

    axs.set_ylabel('Correlation coefficient')
    axs.set_xticks(x + width, model_names, rotation=45, ha='right')
    axs.set_ylim(top=1)
    axs.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", ncol=2)


    # ax = axs[1]
    # ax.bar(x=np.arange(len(paths)), height=pvalues, tick_label=model_names)
    # ax.set_xticks(ax.get_xticks(), model_names, rotation=45, ha='right')
    # # ax.set_yscale('log')

    fig.tight_layout()
    fig.show()
    fig.savefig('figures/correlations.pdf')

    print("")


def plot_ECE(paths, eps=4, figname='ECE'):

    fig, axs = viz.create_figure(ncols=1, figsize=(10, 10))

    model_names = []
    ece_clean_list = []
    ece_adv_list = []

    for path in paths:
        path_splitted = path.split('/')
        model_names.append(path_splitted[-3])
        eps_to_adv_results_dict = get_results(*path_splitted)

        unc_metric = 'entropy_of_mean'

        metric_clean = eps_to_adv_results_dict[0][unc_metric]
        metric_adv = eps_to_adv_results_dict[eps][unc_metric]

        y_true = eps_to_adv_results_dict[0]['ground_truth']
        out_probs = eps_to_adv_results_dict[0]['mean_probs']
        ece_clean, *_ = compute_ece(y_true, out_probs)

        y_true = eps_to_adv_results_dict[eps]['ground_truth']
        out_probs = eps_to_adv_results_dict[eps]['mean_probs']
        ece_adv, *_ = compute_ece(y_true, out_probs)

        ece_clean_list.append(ece_clean)
        ece_adv_list.append(ece_adv)


    width = 0.3
    multiplier = 0
    measures_list = [ece_clean_list, ece_adv_list]
    measure_names = ['ECE clean', 'ECE adv']
    x = np.arange(len(model_names))

    for i, (measure, measure_name) in enumerate(zip(measures_list, measure_names)):
        # if i>0:
        #     break
        offset = width * multiplier
        rects = axs.bar(x + offset, measure, width, label=measure_name)
        # axs.bar_label(rects, padding=3)
        multiplier += 1

    axs.set_xticks(x + width, model_names, rotation=45, ha='right')
    axs.set_ylim(top=1)
    axs.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", ncol=2)


    # ax = axs[1]
    # ax.bar(x=np.arange(len(paths)), height=pvalues, tick_label=model_names)
    # ax.set_xticks(ax.get_xticks(), model_names, rotation=45, ha='right')
    # # ax.set_yscale('log')

    fig.tight_layout()
    fig.suptitle(figname)

    fig.show()
    fig.savefig(f'figures/{figname}.pdf')

    print("")


def plot_cal_curves(paths, ncols=4, figsize=(5, 5), figname='calibration_curves'):
    nplots = len(paths)
    ncols = 5 if nplots == 5 else ncols
    ncols = min(ncols, nplots)
    nrows = nplots // ncols + int(nplots % ncols != 0)
    fig, axs = viz.create_figure(nrows, ncols, figsize=figsize, fontsize=15, squeeze=False)

    model_names = []
    ece_clean_list = []
    ece_adv_list = []

    for plot_i, path in enumerate(paths):
        eps = 4 if 'imagenet' in path else 8
        i, j = plot_i // ncols, plot_i % ncols

        ax = axs[i, j]

        if j == 0:
            ax.set_ylabel('fraction of correct predictions')


        ax.plot([0, 1], [0, 1], linestyle='dashed', color='grey')
        path_splitted = path.split('/')
        eps_to_adv_results_dict = get_results(*path_splitted)

        # print(path)
        # print_pairwise_distance(eps_to_adv_results_dict)

        model_name = path_splitted[-3 if 'semi_robust' in path else -4]
        model_names.append(model_name)

        # CLEAN CURVE
        y_true = eps_to_adv_results_dict[0]['ground_truth']
        out_probs = eps_to_adv_results_dict[0]['mean_probs']
        clean_acc = (y_true.numpy() == out_probs.numpy().argmax(axis=1)).mean()
        clean_ece, clean_bucket_accs, clean_bucket_sizes, clean_avg_conf_list, bin_lowers, bin_uppers, avg_acc_conf = compute_ece(y_true, out_probs)
        # clean_bucket_accs[np.isnan(clean_bucket_accs)] = 0
        ax.plot(clean_avg_conf_list, clean_bucket_accs, label=f"clean (ECE={clean_ece:.3f})", marker='o', color=COLORS[0])
        clean_bucket_sizes = clean_bucket_sizes / clean_bucket_sizes.sum()
        ax.fill_between(clean_avg_conf_list, 0, clean_bucket_sizes, alpha=0.3, color=COLORS[0])
        ax.plot(clean_avg_conf_list, clean_bucket_sizes, alpha=0.5, linestyle='dashed', marker='o', color=COLORS[0])

        print(f"CLEAN {model_name=}, {avg_acc_conf=}")


        # OVER CURVE
        y_true = eps_to_adv_results_dict[eps]['ground_truth']
        out_probs = eps_to_adv_results_dict[eps]['mean_probs']
        rob_acc = (y_true.numpy() == out_probs.numpy().argmax(axis=1)).mean()
        adv_ece, adv_bucket_accs, adv_bucket_sizes, adv_avg_conf_list, bin_lowers, bin_uppers, avg_acc_conf = compute_ece(y_true, out_probs)
        ax.plot(adv_avg_conf_list, adv_bucket_accs, label=f"adv-O (ECE={adv_ece:.3f})", marker='o', color=COLORS[1])
        adv_bucket_sizes = adv_bucket_sizes / adv_bucket_sizes.sum()
        ax.fill_between(adv_avg_conf_list, 0, adv_bucket_sizes, alpha=0.3, color=COLORS[1])
        ax.plot(adv_avg_conf_list, adv_bucket_sizes, alpha=0.5, linestyle='dashed', marker='o', color=COLORS[1])

        print(f"Stab {model_name=}, {avg_acc_conf=}")


        # UNDER CURVE
        eps_to_adv_results_dict = get_results(*path_splitted[:-2], 'U-atk', 'Shake')
        y_true = eps_to_adv_results_dict[eps]['ground_truth']
        out_probs = eps_to_adv_results_dict[eps]['mean_probs']
        urob_acc = (y_true.numpy() == out_probs.numpy().argmax(axis=1)).mean()
        adv_ece, adv_bucket_accs, adv_bucket_sizes, adv_avg_conf_list, bin_lowers, bin_uppers, avg_acc_conf = compute_ece(y_true, out_probs)
        ax.plot(adv_avg_conf_list, adv_bucket_accs, label=f"adv-U (ECE={adv_ece:.3f})", marker='o', color=COLORS[2])
        adv_bucket_sizes = adv_bucket_sizes / adv_bucket_sizes.sum()
        ax.fill_between(adv_avg_conf_list, 0, adv_bucket_sizes, alpha=0.3, color=COLORS[2])
        ax.plot(adv_avg_conf_list, adv_bucket_sizes, alpha=0.5, linestyle='dashed', marker='o', color=COLORS[2])

        print(f"Shake {model_name=}, {avg_acc_conf=}")


        title = f"{model_name}\nacc (clean / advO / advU)\n({clean_acc:.3f} / {rob_acc:.3f} / {urob_acc:.3f})"
        ax.set_title(title)

        # ax.set_xticks([0] + list(bin_lowers))
        # ax.set_yticks([0] + list(bin_lowers))
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.grid('on')
        ax.legend(loc='upper left')

        ax.set_xlim(0, 1)
        # ax.set_ylim(0, adv_bucket_sizes.sum())
        ax.set_ylim(0, 1)
        ax.grid('on')

    # ax = axs[1]
    # ax.bar(x=np.arange(len(paths)), height=pvalues, tick_label=model_names)
    # ax.set_xticks(ax.get_xticks(), model_names, rotation=45, ha='right')
    # # ax.set_yscale('log')
    # axs[0, 0].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", ncol=4)
    for col in range(ncols):
        axs[-1, col].set_xlabel('predicted probability')
    fig.suptitle(figname)
    fig.tight_layout()
    fig.show()
    fig.savefig(f"figures/{figname}.pdf")

    print("")


def plot_conf_displacement(paths, ncols=4, figname = "conf_displacement"):
    figsize = (10, 10)
    nplots = len(paths)
    ncols = min(ncols, nplots)
    nrows = nplots // ncols + int(nplots % ncols != 0)
    fig, axs = viz.create_figure(nrows, ncols, figsize=figsize, fontsize=15, squeeze=False)

    model_names =[]

    for path_i, path in enumerate(paths):
        row, col = path_i // ncols, path_i % ncols

        ax = axs[row, col]
        path_splitted = path.split('/')

        model_name = path_splitted[-3 if 'semi_robust' in path else -4]
        model_names.append(model_name)

        eps_to_adv_results_dict = get_results(*path_splitted)

        eps = 4 if 'imagenet' in path else 8
        y = eps_to_adv_results_dict[0]['ground_truth']
        clean_conf = eps_to_adv_results_dict[0]['mean_probs'].max(axis=1)[0]
        adv_conf = eps_to_adv_results_dict[eps]['mean_probs'].max(axis=1)[0]

        lower_bins = np.arange(0, 1, 0.1)
        upper_bins = lower_bins + 0.1
        conf_matrix = np.zeros(shape=(lower_bins.size, lower_bins.size))
        bin_list = []
        for i, (lower_bin, upper_bin) in enumerate(zip(lower_bins, upper_bins)):
            current_mask = (clean_conf > lower_bin) & (clean_conf < upper_bin)
            for j, (adv_lower_bin, adv_upper_bin) in enumerate(zip(lower_bins, upper_bins)):
                if not (current_mask.sum().item() == 0):
                    adv_mask = (adv_conf[current_mask] > adv_lower_bin) & (adv_conf[current_mask] < adv_upper_bin)
                    conf_matrix[i, j] = adv_mask.sum().item()
            n_samples_in_bin = conf_matrix[i, :].sum()
            if n_samples_in_bin != 0:
                conf_matrix[i, :] /= n_samples_in_bin    # Normalize each row


        sns.heatmap(conf_matrix, annot=True, ax=ax,
                    cmap='summer', cbar=False,  vmin=0, vmax=1, fmt='.3f',
                    xticklabels=[f"{s:.1f}" for s in upper_bins], yticklabels=[f"{s:.1f}" for s in upper_bins])

        ax.set_ylabel('clean bucket')
        ax.set_xlabel('adversarial bucket')

        title = f"{model_name}"
        ax.set_title(title)

    fig.tight_layout()
    fig.show()
    fig.savefig(f"figures/{figname}.pdf")


    # sns.heatmap(conf_matrix, annot=True, fmt='.2f', ax=ax,
    #             cmap='summer', cbar=False,  # vmin=0, vmax=1,
    #             xticklabels=upper_bins, yticklabels=upper_bins)


    print("")


def robust_acc_vs_robust_unc(paths):
    eps = 4
    fig, ax = viz.create_figure(figsize=(10, 10))
    for path1, path2 in zip(paths[0], paths[1]):
        stab_results = get_results(*path1.split('/'))
        autotarget_results = get_results(*path2.split('/'))
        clean_results = stab_results[0]
        stab_results = stab_results[4]
        autotarget_results = autotarget_results[4]


        res = clean_results
        ypred = res['mean_probs']
        print("")


def plot_ECE_gap():
    # atk_type_and_name_list = ['O-atk/Stab']
    # atk_type_and_name_list = ['U-atk/Shake']
    datasets = ['cifar10', 'imagenet']
    # datasets = ['imagenet']
    robusts = [True]

    dict_terms = {'semi_robust': 'robust',
                  'naive_robust': 'naive'}

    # robust_acc_vs_robust_unc(paths)
    # stab_path_list = [item for path_list in stab_path_list for item in path_list]
    for atk_type_and_name_list in [['O-atk/Stab'], ['U-atk/Shake']]:
        path_lists = get_paths(datasets, atk_type_and_name_list, robusts)
        atk_name = atk_type_and_name_list[0].split('/')[0]
        for stab_path_list in path_lists:
            robustness_level = dict_terms[stab_path_list[0].split('/')[2]]
            ds_name = stab_path_list[0].split('/')[3]


            ata_path_list = [path.replace('Stab', 'AutoTarget') for path in stab_path_list]

            # robust_accuracy = []
            # for path in ata_path_list:
            #     eps = 8 if 'cifar' in path else 4
            #     res = get_results(*path.split('/'))
            #     y = res[eps]['ground_truth']
            #     ypred = res[eps]['mean_probs'].argmax(axis=1)
            #     robacc = (ypred == y).numpy().mean()
            #     robust_accuracy.append(robacc)

            uncertainty_before_atk = []
            uncertainty_after_atk = []
            model_names = []
            for path in stab_path_list:
                path_splitted = path.split('/')

                model_names.append(path_splitted[path_splitted.index('None') + (1 if not 'naive' in path else -1)])
                eps = 8 if 'cifar' in path else 4
                res = get_results(*path.split('/'))

                y = res[0]['ground_truth']
                probs = res[0]['mean_probs']
                ece = compute_ece(y_true=y, out_probs=probs, compute_abs=False)[0]
                uncertainty_before_atk.append(ece)

                y = res[eps]['ground_truth']
                probs = res[eps]['mean_probs']
                ece = compute_ece(y_true=y, out_probs=probs, compute_abs=False)[0]
                uncertainty_after_atk.append(ece)

            fig, ax = viz.create_figure(figsize=(15, 15))

            for i, mname in enumerate(model_names):
                ax.scatter(uncertainty_before_atk[i], uncertainty_after_atk[i] - uncertainty_before_atk[i],
                           color=ALL_COLORS[i], marker=ALL_MARKERS[i], alpha=0.8, s=400, label=mname)
                # ax.scatter(uncertainty_before_atk[i], robust_accuracy[i],
                #            color=ALL_COLORS[i], marker='', alpha=0.4, s=200)
                # ax.plot([uncertainty_before_atk[i], uncertainty_after_atk[i]],
                #         [robust_accuracy[i], robust_accuracy[i]], color=ALL_COLORS[i], linestyle='dashed')


            # title = f"ECE_GAP_{atk_name}_{robustness_level}_{ds_name}_with_clean_results"
            title = f"ECE_GAP_{atk_name}_{robustness_level}_{ds_name}"
            ax.set_title(title)
            ax.set_xlabel('signed ECE')
            ax.set_ylabel('GAP after - before attack')
            ax.legend()
            fig.show()
            fig.savefig(f"figures/{title}.pdf")
            print("")
    print("")


def plot_entropy_gap():
    # atk_type_and_name_list = ['O-atk/Stab']
    # atk_type_and_name_list = ['U-atk/Shake']
    datasets = ['cifar10', 'imagenet']
    # datasets = ['imagenet']
    robusts = [True]

    dict_terms = {'semi_robust': 'robust',
                  'naive_robust': 'naive'}

    # robust_acc_vs_robust_unc(paths)
    # stab_path_list = [item for path_list in stab_path_list for item in path_list]
    for atk_type_and_name_list in [['O-atk/Stab'], ['U-atk/Shake']]:
        path_lists = get_paths(datasets, atk_type_and_name_list, robusts)
        atk_name = atk_type_and_name_list[0].split('/')[0]
        for stab_path_list in path_lists:
            robustness_level = dict_terms[stab_path_list[0].split('/')[2]]
            ds_name = stab_path_list[0].split('/')[3]


            ata_path_list = [path.replace('Stab', 'AutoTarget') for path in stab_path_list]

            # robust_accuracy = []
            # for path in ata_path_list:
            #     eps = 8 if 'cifar' in path else 4
            #     res = get_results(*path.split('/'))
            #     y = res[eps]['ground_truth']
            #     ypred = res[eps]['mean_probs'].argmax(axis=1)
            #     robacc = (ypred == y).numpy().mean()
            #     robust_accuracy.append(robacc)

            uncertainty_before_atk = []
            uncertainty_after_atk = []
            model_names = []
            for path in stab_path_list:
                path_splitted = path.split('/')

                model_names.append(path_splitted[path_splitted.index('None') + (1 if not 'naive' in path else -1)])
                eps = 8 if 'cifar' in path else 4
                res = get_results(*path.split('/'))

                uncertainty_before_atk.append(res[0]['entropy_of_mean'].mean().item())
                uncertainty_after_atk.append(res[eps]['entropy_of_mean'].mean().item())

            fig, ax = viz.create_figure(figsize=(15, 15))

            for i, mname in enumerate(model_names):
                ax.scatter(uncertainty_before_atk[i], uncertainty_after_atk[i] - uncertainty_before_atk[i],
                           color=ALL_COLORS[i], marker=ALL_MARKERS[i], alpha=0.8, s=400, label=mname)
                # ax.scatter(uncertainty_before_atk[i], robust_accuracy[i],
                #            color=ALL_COLORS[i], marker='', alpha=0.4, s=200)
                # ax.plot([uncertainty_before_atk[i], uncertainty_after_atk[i]],
                #         [robust_accuracy[i], robust_accuracy[i]], color=ALL_COLORS[i], linestyle='dashed')


            # title = f"ECE_GAP_{atk_name}_{robustness_level}_{ds_name}_with_clean_results"
            title = f"ENTROPY_GAP_{atk_name}_{robustness_level}_{ds_name}"
            ax.set_title(title)
            ax.set_xlabel('H(x)')
            ax.set_ylabel('H(x*) - H(x)')
            ax.legend()
            fig.show()
            fig.savefig(f"figures/{title}.pdf")
            print("")
    print("")


def plot_conf_theory():
    # atk_type_and_name_list = ['O-atk/Stab']
    # atk_type_and_name_list = ['U-atk/Shake']
    datasets = ['cifar10', 'imagenet']
    # datasets = ['imagenet']
    robusts = [True]

    dict_terms = {'semi_robust': 'robust',
                  'naive_robust': 'naive'}

    # robust_acc_vs_robust_unc(paths)
    # stab_path_list = [item for path_list in stab_path_list for item in path_list]
    for atk_type_and_name_list in [['O-atk/Stab'], ['U-atk/Shake']]:
        path_lists = get_paths(datasets, atk_type_and_name_list, robusts)
        atk_name = atk_type_and_name_list[0].split('/')[0]
        for stab_path_list in path_lists:
            robustness_level = dict_terms[stab_path_list[0].split('/')[2]]
            ds_name = stab_path_list[0].split('/')[3]

            ata_path_list = [path.replace('Stab', 'AutoTarget') for path in stab_path_list]

            uncertainty_before_atk = []
            uncertainty_after_atk = []
            model_names = []
            for path in stab_path_list:
                path_splitted = path.split('/')

                model_names.append(path_splitted[path_splitted.index('None') + (1 if not 'naive' in path else -1)])
                eps = 8 if 'cifar' in path else 4
                res = get_results(*path.split('/'))


                y = res[0]['ground_truth']
                preds = res[0]['preds']
                probs = res[0]['mean_probs']


                y = res[eps]['ground_truth']
                preds = res[eps]['preds']
                probs = res[eps]['mean_probs']

            fig, ax = viz.create_figure(figsize=(15, 15))

            for i, mname in enumerate(model_names):
                ax.scatter(uncertainty_after_atk[i], robust_accuracy[i],
                           color=ALL_COLORS[i], marker=ALL_MARKERS[i], alpha=0.8, s=400, label=mname)
                ax.scatter(uncertainty_before_atk[i], robust_accuracy[i],
                           color=ALL_COLORS[i], marker='', alpha=0.4, s=200)
                ax.plot([uncertainty_before_atk[i], uncertainty_after_atk[i]],
                        [robust_accuracy[i], robust_accuracy[i]], color=ALL_COLORS[i], linestyle='dashed')


            title = f"SUMUP_{atk_name}_{robustness_level}_{ds_name}_with_clean_results"
            # title = f"SUMUP_{atk_name}_{robustness_level}_{ds_name}"
            ax.set_title(title)
            ax.set_xlabel('signed ECE')
            ax.set_ylabel('Robust Accuracy')
            ax.legend()
            fig.show()
            fig.savefig(f"figures/{title}.pdf")
            print("")
    print("")


def plot_sumup():
    # atk_type_and_name_list = ['O-atk/Stab']
    # atk_type_and_name_list = ['U-atk/Shake']
    datasets = ['cifar10', 'imagenet']
    # datasets = ['imagenet']
    robusts = [True]

    dict_terms = {'semi_robust': 'robust',
                  'naive_robust': 'naive'}

    # robust_acc_vs_robust_unc(paths)
    # stab_path_list = [item for path_list in stab_path_list for item in path_list]
    for atk_type_and_name_list in [['O-atk/Stab'], ['U-atk/Shake']]:
        path_lists = get_paths(datasets, atk_type_and_name_list, robusts)
        atk_name = atk_type_and_name_list[0].split('/')[0]
        for stab_path_list in path_lists:
            robustness_level = dict_terms[stab_path_list[0].split('/')[2]]
            ds_name = stab_path_list[0].split('/')[3]


            ata_path_list = [path.replace('Stab', 'AutoTarget') for path in stab_path_list]

            robust_accuracy = []
            for path in ata_path_list:
                eps = 8 if 'cifar' in path else 4
                res = get_results(*path.split('/'))
                y = res[eps]['ground_truth']
                ypred = res[eps]['mean_probs'].argmax(axis=1)
                robacc = (ypred == y).numpy().mean()
                robust_accuracy.append(robacc)

            uncertainty_before_atk = []
            uncertainty_after_atk = []
            model_names = []
            for path in stab_path_list:
                path_splitted = path.split('/')

                model_names.append(path_splitted[path_splitted.index('None') + (1 if not 'naive' in path else -1)])
                eps = 8 if 'cifar' in path else 4
                res = get_results(*path.split('/'))


                y = res[0]['ground_truth']
                probs = res[0]['mean_probs']
                ece = compute_ece(y_true=y, out_probs=probs, compute_abs=False)[0]
                uncertainty_before_atk.append(ece)

                y = res[eps]['ground_truth']
                probs = res[eps]['mean_probs']
                ece = compute_ece(y_true=y, out_probs=probs, compute_abs=False)[0]
                uncertainty_after_atk.append(ece)

            fig, ax = viz.create_figure(figsize=(15, 15))

            for i, mname in enumerate(model_names):
                ax.scatter(uncertainty_after_atk[i], robust_accuracy[i],
                           color=ALL_COLORS[i], marker=ALL_MARKERS[i], alpha=0.8, s=400, label=mname)
                ax.scatter(uncertainty_before_atk[i], robust_accuracy[i],
                           color=ALL_COLORS[i], marker='', alpha=0.4, s=200)
                ax.plot([uncertainty_before_atk[i], uncertainty_after_atk[i]],
                        [robust_accuracy[i], robust_accuracy[i]], color=ALL_COLORS[i], linestyle='dashed')


            title = f"SUMUP_{atk_name}_{robustness_level}_{ds_name}_with_clean_results"
            # title = f"SUMUP_{atk_name}_{robustness_level}_{ds_name}"
            ax.set_title(title)
            ax.set_xlabel('signed ECE')
            ax.set_ylabel('Robust Accuracy')
            ax.legend()
            fig.show()
            fig.savefig(f"figures/{title}.pdf")
            print("")
    print("")


def plot_cal_curves_and_hist(paths, figsize=(5, 5), figtitle='calibration_curves'):
    """
    Prende le paths degli attacchi Oatk e aggiunge gli Uatk. Vengono generati un reliability ed un histogram
    per ogni modello e salvati.
    """

    save_path = "figures/calibration_hist/"

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for plot_i, path in enumerate(paths):
        eps = 4 if 'imagenet' in path else 8
        fig, axs = viz.create_figure(2, 1, figsize=figsize, fontsize=15, squeeze=False)

        path_splitted = path.split('/')
        eps_to_adv_results_dict = get_results(*path_splitted)

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

        print(f"Stab {model_name=}, {avg_acc_conf=}")


        # UNDER CURVE
        axs[0,0].set_ylabel('fraction of correct predictions')
        axs[1, 0].set_ylabel('Number of samples')
        axs[0,0].plot([0, 1], [0, 1], linestyle='dashed', color='grey')
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
        # plot_cal_curves_and_hist(path_list, figsize=(5,10), figtitle=title)

    # ------------------------ CONFIDENCE DISPLACEMENT PLOTS
    # Non so cosa fa
    # for path_list in paths:
    #     robustness_level = dict_terms[path_list[0].split('/')[2]]
    #     ds_name = path_list[0].split('/')[3]
    #     attack_type = path_list[0].split('/')[-1]
    #     title = f"conf_displacement_{robustness_level}_{ds_name}_{attack_type}"
    #     plot_conf_displacement(path_list, figname=title)


# ####################################### OLD MAIN ###########################################################
#     # plot_all_scatters(paths, metric='entropy_of_mean')
#     # plot_correlations(paths)
#     # plot_ECE(paths)
#     # plot_cal_curves(paths)
#     """
#     >>> ROBUST ACCURACY -> ROBUST UNCERTAINTY
#     - difesa tradizionale funziona? Si
#     ----> mostrare transformer calibration curve naive e robust (tutti gli altri in appendix completi)
#     - Come correla robustezza a predizioni e robustezza a incertezza?
#     ----> mostrare scatter plot tra robust uncertainty e robust accuracy:
#           come misura puntuale di uncertainty usare la ECE senza valore assoluto
#
#     - Risk management: underconf / overconf implica migliore calibrazione sotto attacchi O-atk / U-atk
#     - H(R) > H(Q) -> D(R) > D(x)
#
#     - displacement softmax before/after attack -> higher robustness (forse banale)
#     - dati clean e autotarget: vedere distribuzione entropia tra sample corretti e misclassificati
#
#
#     PLOTS:
#     - clean VS adversarial entropy, con clean entropy ordinate
#     -
#     """
#
#
#
#
#     print("")








