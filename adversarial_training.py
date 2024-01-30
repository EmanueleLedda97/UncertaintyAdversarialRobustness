import utils.utils
from models.utils import load_model
from attacks.bayesian_attacks import StabilizingAttack, AutoTargetAttack
from attacks.loss_functions import UncertaintyDivergenceLoss
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
import os
import pickle
from tqdm import tqdm

import metrics

import attacks

import evaluation as eval

import reliability_diagram


from utils.adv_train_parser import parse_main_classification, add_ood_parsing


# --------------------
def val_set_evaluation(model, validation_dataloader, dict_metrics, \
                        device, attack, num_attack_iterations, mc_samples_attack, loss_adv_training):
        
        first_batch = next(iter(validation_dataloader))
        for batch_i, (x, y) in enumerate([first_batch]):
        # for batch_i, (x, y) in enumerate(validation_dataloader):
            # Sending the data to device
            x, y = x.to(device), y.to(device)                  
            
            # Computing the adv example
            x_adv = attack.run(x=x, y=y, iterations=num_attack_iterations)

            model.eval()
            with torch.no_grad():    
                # Getting the loss path
                loss_path = attack.loss_fn.loss_path['CE']
                attack.loss_fn.loss_path['CE'] = []

                # Populating the loss paths
                dict_metrics["attack_loss_start_path"].append(loss_path[0])
                dict_metrics["attack_loss_end_path"].append(loss_path[-1])
                dict_metrics["attack_gap_path"].append(loss_path[0] - loss_path[-1])
                print(f"validation efficacy: start {loss_path[0]} end {loss_path[-1]}; delta {loss_path[0]-loss_path[-1]}")

                clean_out = model(x, mc_sample_size=mc_samples_attack)
                adv_out = model(x_adv, mc_sample_size=mc_samples_attack)

                # Computing the loss
                loss = loss_adv_training(clean_output=clean_out, adv_output=adv_out, target=y)

                dict_metrics["model_loss_path"].append(loss.item())

                acc = (((F.softmax(clean_out, dim = 1).argmax(dim=1) == y).sum() / x.size(0)) * 100).item()
                # print(acc)
                dict_metrics["accuracy_path"].append(acc)

                all_metrics = eval.evaluate_batch_bayesian(model, x, y, mc_sample_size=mc_samples_attack)
            
                acc = (((all_metrics["ground_truth"] == all_metrics["preds"]).sum() / x.size(0)) * 100).item()
            
            
            return 

def print_metrics(dict_metrics, fig_path = None):
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))

    axes[0,0].plot(dict_metrics["attack_loss_start_path"])
    axes[0,0].plot(dict_metrics["attack_loss_end_path"])
    axes[0,0].legend(['start', 'end'])
    axes[0,1].plot(dict_metrics["attack_gap_path"])
    axes[1,0].plot(dict_metrics["accuracy_path"])
    axes[1,1].plot(dict_metrics["model_loss_path"])

    axes[0,0].set_title('Attack Loss')
    axes[0,1].set_title('Atk Loss Gap')
    axes[1,0].set_title('Accuracy')
    axes[1,1].set_title('Model Loss')

    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path)

# -----------------------------

# parameters = {
#     "cuda" : 0,

#     # DATASET AND MODEL SETTING
#     "dataset" : 'cifar10',
#     "backbone" : 'resnet18',
#     "uq_technique" : 'embedded_dropout',
#     "dropout_rate" : 0.1,

#     # ATTACK SETTINGS
#     "epsilon" : 8/255,
#     "attack_update_strategy":'pgd',
#     "step_size":5e-3,
#     "mc_samples_attack" : 5,
#     "attack_iterations": 1,

#     # TRAIN SETTINGS
#     "batch_size" : 256,           # USING 2000
#     "epochs" : 10,
#     "lr" : 1e-4,                 # 5e-4 usato from scratch, 1e-6 / 1e-7 per finetuning
#     "alpha": 1,
#     "beta" : 3    # OCCHIO che usavo sempre 2 e 8
# }

def main(root="adv_exp",
         experiment_type='classification_id',
        #  robustness_level='naive_robust',
         dataset='cifar10',
         backbone='resnet18',
         uq_technique='injected_dropout',
         dropout_rate=None,
         
         attack_loss='Stab',
         attack_update_strategy='pgd',
        #  norm='Linf',
         num_attack_iterations=10,

         
         mc_samples_attack=5,
         epsilon=(8/255),
         step_size=None,


         batch_size=256,                    
         epochs = 10,
         lr = 1e-4,
         alpha = 1,
         beta = 10,

        #  mc_samples_eval=100,
        #  batch_size_eval=100,

        #  re_evaluation_mode=False,  # TODO: Add to dynamic argument selection when parsing
         full_bayesian=True,        # TODO: Add to dynamic argument selection when parsing

        #  ood_dataset=None,
        #  iid_size=None,
        #  ood_size=None,

         seed=0,
         cuda=0,
         kwargs=None,):

    is_an_ood_experiment = (experiment_type == 'classification_ood')
    # attack_parameters = (epsilon, norm, attack_update_strategy, step_size, mc_samples_attack)

    device = utils.utils.get_device(cuda) if torch.cuda.is_available() else 'cpu'
    model = load_model(backbone, uq_technique,                              # Loading the model
                       dataset, transform=utils.utils.get_normalizer(dataset),
                       dropout_rate=dropout_rate,
                       full_bayesian=full_bayesian,
                       device=device)
    model.to(device)  

    attack_kwargs = {'model': model,
                        'device': device,
                        'epsilon': epsilon,
                        'update_strategy': attack_update_strategy,
                        'step_size': step_size}
    
    if attack_loss == 'MinVar':
        attack = attacks.bayesian_attacks.MinVarAttack(mc_sample_size_during_attack=mc_samples_attack, **attack_kwargs)
    elif attack_loss == 'MaxVar':
        attack = attacks.bayesian_attacks.MaxVarAttack(mc_sample_size_during_attack=mc_samples_attack, **attack_kwargs)
    elif attack_loss == 'AutoTarget':
        attack = attacks.bayesian_attacks.AutoTargetAttack(mc_sample_size_during_attack=mc_samples_attack, **attack_kwargs)
    elif attack_loss == 'Stab':
        attack = attacks.bayesian_attacks.StabilizingAttack(mc_sample_size_during_attack=mc_samples_attack, **attack_kwargs)
    elif attack_loss == 'Centroid':
        attack = attacks.bayesian_attacks.DUQAttack(**attack_kwargs)
    else:
        raise Exception(attack_loss, "attack loss is not supported.")



    training_metrics = {
        "attack_loss_start_path" : [],
        "attack_loss_end_path" : [],
        "attack_gap_path" : [],
        "accuracy_path" : [],
        "model_loss_path" : []
    }

    validation_metrics = {
        "attack_loss_start_path" : [],
        "attack_loss_end_path" : [],
        "attack_gap_path" : [],
        "accuracy_path" : [],
        "model_loss_path" : []
    }


    train_set, validation_set, test_set = utils.utils.get_dataset_splits(dataset=dataset,
                                                                        set_normalization=False,
                                                                        ood=False,
                                                                        load_adversarial_set=False)


    train_subset_loader_during_attack = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=16)
    validation_dataloader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=16)

    optimizer = torch.optim.RMSprop(model.parameters(),lr=lr)
    loss_adv_training = UncertaintyDivergenceLoss(alpha=alpha, beta=beta)


# ------------------------------------- METRICS PRE TRAINING
    model.train()
    all_train_metrics = eval.evaluate_bayesian(model, train_subset_loader_during_attack, mc_sample_size=mc_samples_attack, device=device, seed=0)
    
    # pred, unc = metrics.get_prediction_with_uncertainty(all_train_metrics["mean_probs"])
    # ece = metrics.expected_calibration_error(pred, unc, all_train_metrics["ground_truth"], bins=10)
    # print(f"{ece=}")

    calibration = reliability_diagram.compute_calibration(all_train_metrics["ground_truth"].numpy(),
                                            all_train_metrics["preds"].numpy(),
                                            all_train_metrics["mean_probs"].max(1)[0].numpy())
    
    print(calibration)

    fig = reliability_diagram.reliability_diagram(all_train_metrics["ground_truth"].numpy(),
                                            all_train_metrics["preds"].numpy(),
                                            all_train_metrics["mean_probs"].max(1)[0].numpy(), 
                                            return_fig=True)

    fig.savefig("rel_plot.png", format = "png")


# -------------------------------- TRAIN ---------------------------------------------------------------
    
    validation_args = {"device":device, 
                       "attack" : attack, 
                       "num_attack_iterations" : num_attack_iterations, 
                       "mc_samples_attack": mc_samples_attack, 
                       "loss_adv_training" : loss_adv_training}

    first_batch = next(iter(train_subset_loader_during_attack))
    for e in tqdm(range(epochs), desc="EPOCHS"):
        # Generating iteratively the adversarial examples from the selected test set
        # for batch_i, (x, y) in enumerate(tqdm(train_subset_loader_during_attack, desc="BATCHES")):
        for batch_i, (x, y) in enumerate([first_batch]):
            model.train()
            
            # Sending the data to device
            x, y = x.to(device), y.to(device)                  
            
            # Computing the adv example
            x_adv = attack.run(x=x, y=y, iterations=num_attack_iterations)
            #optimizer.zero_grad()
            
            # Getting the loss path
            loss_path = attack.loss_fn.loss_path['CE']
            attack.loss_fn.loss_path['CE'] = []

            # Populating the loss paths
            training_metrics["attack_loss_start_path"].append(loss_path[0])
            training_metrics["attack_loss_end_path"].append(loss_path[-1])
            training_metrics["attack_gap_path"].append(loss_path[0] - loss_path[-1])
            # print(f"Attack efficacy: start {loss_path[0]} end {loss_path[-1]}; delta {loss_path[0]-loss_path[-1]}")

            optimizer.zero_grad()

            clean_out = model(x, mc_sample_size=mc_samples_attack)
            adv_out = model(x_adv, mc_sample_size=mc_samples_attack)

            # Computing the loss
            loss = loss_adv_training(clean_output=clean_out, adv_output=adv_out, target=y)
            # print(f"{loss.item()=}")
            training_metrics["model_loss_path"].append(loss.item())

            all_metrics_clean = eval.evaluate_batch_bayesian(model, x, y, mc_sample_size=mc_samples_attack)
            # all_metrics_adv = eval.evaluate_batch_bayesian(model, x_adv, y, mc_sample_size=mc_samples_attack) # ha senso calcolare metriche per adv?

        
            acc = (((all_metrics_clean["ground_truth"] == all_metrics_clean["preds"]).sum() / x.size(0)) * 100).item()
            # print(acc)
            training_metrics["accuracy_path"].append(acc)

            # Backpropagation
            loss.backward()
            optimizer.step()


            # if batch_i%25 == 0:
                # print(f"{batch_i=}, {loss.item()=}")
                # print_metrics(training_metrics, fig_path = f'{OUTPUT_FOLDER}EXPERIMENT---aggregate_losses.png')
                # print_metrics(training_metrics, fig_path = f'{OUTPUT_FOLDER}EXPERIMENT---aggregate_losses.png')
                # torch.save(model.state_dict(), f'{OUTPUT_FOLDER}EXPERIMENT---model-weights.pt')
                # with open(f'{OUTPUT_FOLDER}EXPERIMENT---metrics_dict.pkl', "wb") as f:
                #     pickle.dump(training_metrics, f)



            # if batch_i%20 == 0:
        val_set_evaluation(model ,validation_dataloader, validation_metrics, **validation_args)

    # print_metrics(training_metrics, fig_path = f'{OUTPUT_FOLDER}EXPERIMENT---aggregate_losses.png')
    # torch.save(model.state_dict(), f'{OUTPUT_FOLDER}EXPERIMENT---model-weights.pt')

    # with open(f'{OUTPUT_FOLDER}EXPERIMENT---metrics_dict.pkl', "wb") as f:
    #     pickle.dump(training_metrics, f)


def main_single(parser):

    # Parsing the arguments
    args = parser.parse_args()
    kwargs = {key: value for (key, value) in args._get_kwargs()}

    # Running the main
    main(**kwargs, kwargs=kwargs)


if __name__ == '__main__':
    
    # Creating the parser for the script arguments
    parser = parse_main_classification()
    
    # Parsing and running the main with the seceval mode
    args = parser.parse_args()
    if args.experiment_type == 'classification_ood':
        parser = add_ood_parsing(parser)

    main_single(parser)
