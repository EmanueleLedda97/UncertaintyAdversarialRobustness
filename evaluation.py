import torch
import metrics
import numpy as np
import utils.utils as ut


'''
    TODO: Anche qua bisogna fare attenzione! Alcune cose sono vecchie implementazioni
'''

METRIC_KEYS = ['ground_truth', 'mean_probs', 'var_probs', 'preds', 'var',
               'mean_of_entropies', 'entropy_of_mean', 'mutual_information', 'true_var']

METRIC_DUQ_KEYS = ['ground_truth', 'preds', 'centroids', 'confidence']

# TODO: implement after implementing Dataset and Dataloader for adversarial examples
def get_dataset_predictions_and_uncertainties(model, data_loader):
    pass


def _eval_duq(model, x, y):

    centroids = model(x)
    ground_truth = y.detach().cpu()
    preds = centroids.argmax(dim=1, keepdim=True).flatten().detach().cpu()
    confidences = torch.max(centroids, dim=1)[0].flatten().detach().cpu()
    partial_results = [ground_truth, preds, centroids, confidences]
    results_dict = {k: v for (k, v) in zip(METRIC_DUQ_KEYS, partial_results)}

    return results_dict

def _eval(model, x, y, mc_sample_size=20):
    """
    It returns in output:

    mean_probs (len(x), 10)
    var_probs (len(x), 10)
    preds -> (len(x),)
    var -> (len(x),)
    entropy -> (len(x),)
    mutual_information -> (len(x),)

    """

    output_logits = model(x, mc_sample_size=mc_sample_size, get_mc_output=True)
    output_probs = torch.nn.Softmax(dim=-1)(output_logits)

    ground_truth = y.detach().cpu()
    mean_probs = metrics.mc_samples_mean(output_probs).detach().cpu()
    var_probs = metrics.mc_samples_var(output_probs).detach().cpu()
    preds = mean_probs.argmax(dim=1, keepdim=True).flatten().detach().cpu()
    var = metrics.var(output_probs).detach().cpu()
    mean_of_entropies = metrics.entropy(output_probs, aleatoric_mode=False).detach().cpu()
    entropy_of_mean = metrics.entropy(output_probs, aleatoric_mode=True).detach().cpu()
    mutual_information = metrics.mutual_information(output_probs).detach().cpu()
    true_var = metrics.true_var(output_probs).detach().cpu()
    
    # ground_truth = y.detach().cpu().numpy()
    # mean_probs = metrics.mc_samples_mean(output_probs)
    # var_probs = metrics.mc_samples_var(output_probs).detach().cpu().numpy()
    # preds = mean_probs.argmax(dim=1, keepdim=True).flatten().detach().cpu().numpy()
    # var = metrics.var(output_probs).detach().cpu().numpy()
    # mean_of_entropies = metrics.entropy(output_probs, aleatoric_mode=False).detach().cpu().numpy()
    # entropy_of_mean = metrics.entropy(output_probs, aleatoric_mode=True).detach().cpu().numpy()
    # mutual_information = metrics.mutual_information(output_probs).detach().cpu().numpy()

    partial_results = [ground_truth, mean_probs, var_probs, preds, var, 
                       mean_of_entropies, entropy_of_mean, mutual_information, true_var]
    results_dict = {k: v for (k, v) in zip(METRIC_KEYS, partial_results)}

    return results_dict

import utils.utils
def evaluate_bayesian(model, test_loader, mc_sample_size=20, device='cpu', seed=0):
    model.to(device)
    model.eval()
    all_results = {k: None for k in METRIC_KEYS}
    utils.utils.set_all_seed(seed)
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        results_dict = _eval(model, x, y, mc_sample_size)
        for k, v in results_dict.items():
            all_results[k] = torch.cat((all_results[k], v), dim=0) if all_results[k] is not None else v

    return all_results


def evaluate_deterministic(model, test_loader, device='cpu', seed=0):
    model.to(device)
    model.eval()
    all_results = {k: None for k in METRIC_DUQ_KEYS}
    ut.set_all_seed(seed)
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        results_dict = _eval_duq(model, x, y)
        for k, v in results_dict.items():
            all_results[k] = torch.cat((all_results[k], v), dim=0) if all_results[k] is not None else v

    return all_results


def evaluate_batch_standard(model, x, y):
    model.eval()
    output = model(x)
    preds = output.argmax(dim=1, keepdim=True)
    correct = preds.eq(y.view_as(preds)).sum().item()
    accuracy = correct / x.shape[0]
    print(f"Accuracy: {accuracy}")
    torch.cuda.empty_cache()
    return



# def _eval(model, x, y, mc_sample_size=20):
#     output = model(x, mc_sample_size=mc_sample_size, get_mc_output=True)
#     output_probs = torch.nn.Softmax(dim=-1)(output)
    
#     mean_probs = metrics.mc_samples_mean(output_probs)
#     preds = mean_probs.argmax(dim=1, keepdim=True)
#     correct = preds.eq(y.cpu().view_as(preds)).sum().item()
#     accuracy = correct / x.shape[0]
#     print(f"Accuracy: {accuracy:.3f}")

#     mi = metrics.mutual_information(output_probs)
#     memory_allocated = f"{torch.cuda.memory_allocated()/1e6} MiB"
#     print(f"MI -> mean:{mi.mean():.3f}")# / std:{mi.std():.3f} / min:{mi.min():.3f} / max:{mi.max():.3f}")

#     if correct_mask is None:
#         correct_mask = (preds.flatten() == y.cpu())
#     if non_correct_mask is None:
#         non_correct_mask = (preds.flatten() != y.cpu())
#     mi_correct = metrics.mutual_information(output_probs[:, correct_mask, :])
#     mi_non_correct = metrics.mutual_information(output_probs[:, non_correct_mask, :])
#     print(f"MI (correct)     -> mean:{mi_correct.mean():.3f}")# / std:{mi_correct.std():.3f}" \
#         #   f"min:{mi_correct.min():.3f} / max:{mi_correct.max():.3f}")
#     print(f"MI (non-correct) -> mean:{mi_non_correct.mean():.3f}")# / std:{mi_non_correct.std():.3f}" \
#         #    f"min:{mi_non_correct.min():.3f} / max:{mi_non_correct.max():.3f}")

#     # print(f"(allocated {memory_allocated})")

#     results = {'acc': accuracy,
#                'mi': mi,
#                'mi_correct': mi_correct,
#                'mi_non_correct': mi_non_correct,
#                'correct_mask': correct_mask,
#                'non_correct_mask': non_correct_mask}
#     return results

def evaluate_batch_bayesian(model, x, y, mc_sample_size=50, correct_mask=None, non_correct_mask=None,
                            logger=None):
    torch.cuda.empty_cache()
    # Magheggio per liberare tutta la memoria direttamente qui
    results = _eval(model, x, y, mc_sample_size, correct_mask, non_correct_mask, logger)
    torch.cuda.empty_cache()
    return results


def seceval(model, x, y, attack, epsilon_max, epsilon_step, attack_iterations, mc_sample_size_during_evaluation=100, device='cpu'):
    for epsilon_i in np.arange(start=0, stop=epsilon_max, step=epsilon_step):
        ######################################################
        # GREYY BOX ATTACKS
        ######################################################

        model.eval()
        x_adv = attack.run(x=x, target=y, model=model.backbone, iterations=attack_iterations, device=device)

        output = model()
        # x_adv = apgd(model, x, y, eps=epsilon, norm=float('inf'), n_iter=attack_iterations)
        # adversary = AutoAttack(model, norm='Linf', eps=8/255)
        # x_adv = adversary.run_standard_evaluation(x, y)

        # print("- Only Backbone:")
        # evaluate_batch_standard(model.backbone, x_adv, y)
        # print("- Bayesian model:")
        # evaluate_batch_bayesian(model, x_adv, y, mc_sample_size=mc_sample_size_during_evaluation)
        # print(f"Attack Took {attack.elapsed_time:.3f} seconds ({attack.elapsed_time/attack_iterations:.3f} per iter)")
        # attack.loss_fn.plot_loss(fig_path='experiments/loss.pdf')
