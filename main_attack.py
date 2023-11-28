import torch
import torchvision
from models.utils import load_model
import evaluation as eval
import utils
import utils.constants as keys
import attacks.standard_attacks
import attacks.bayesian_attacks
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from utils.paths import get_full_experiment_path
from utils.parsers import parse_main_classification, add_seceval_parsing, add_ood_parsing
from utils.loggers import set_up_logger


'''
    /* BASE PARAMETERS */
    backbone: Base network architecture
    uq_technique: Uncertainty Quantification technique we want to use
    dataset: Dataset for the experiments

    TODO: Add full documentation
'''

def main(root=keys.ROOT,
         experiment_type='classification_id',
         robustness_level='naive_robust',
         dataset='cifar10',
         backbone='resnet18',
         uq_technique='injected_dropout',
         dropout_rate=None,
         
         attack_loss='Stab',
         attack_update_strategy='pgd',
         norm='Linf',
         num_attack_iterations=10,
         
         mc_samples_attack=30,
         num_adv_examples=100,
         batch_size=100,            # TODO: Refactor in "attack_batch_size"
         epsilon=(8/255)*5,
         step_size=None,

         mc_samples_eval=100,
         batch_size_eval=100,

         re_evaluation_mode=False,  # TODO: Add to dynamic argument selection when parsing
         full_bayesian=True,        # TODO: Add to dynamic argument selection when parsing

         ood_dataset=None,
         iid_size=None,
         ood_size=None,

         seed=0,
         cuda=0,
         kwargs=None,
         ):

    # Computing utility logic variables
    is_an_ood_experiment = (experiment_type == 'classification_ood')


    # Setting up the experiment paths
    attack_parameters = (epsilon, norm, attack_update_strategy, step_size, mc_samples_attack)
    experiment_path, clean_results_path, adv_results_path = get_full_experiment_path(experiment_type,
                                                                                     robustness_level,
                                                                                     dataset,
                                                                                     backbone,
                                                                                     uq_technique,
                                                                                     attack_loss,
                                                                                     attack_parameters,
                                                                                     dropout_rate=dropout_rate,
                                                                                     ood_dataset=ood_dataset,
                                                                                     iid_size=iid_size,
                                                                                     ood_size=ood_size)
    adv_examples_path = os.path.join(experiment_path, 'generated_adversarial_examples')

    # Creating the experiment's folder
    if not os.path.isdir(adv_examples_path):
        os.makedirs(adv_examples_path)     

    # Setting up the logger
    logger = set_up_logger(experiment_path, cuda, kwargs)
    
    # Loading the model and sending to device
    logger.debug("Loading model...")
    device = utils.get_device(cuda) if torch.cuda.is_available() else 'cpu'
    model = load_model(backbone, uq_technique,                              # Loading the model
                       dataset, transform=utils.utils.get_normalizer(dataset),
                       dropout_rate=dropout_rate,
                       full_bayesian=full_bayesian,
                       device=device)
    
    model.to(device)                                                        # ... and sending the model to device

    # Loading the dataset
    logger.debug("Loading dataset ...")
    test_subset_set = utils.utils.get_dataset_splits(dataset=dataset,               # Loading the 'dataset'... 
                                                     set_normalization=False,       # ... without normalization due to transform in forward... 
                                                     ood=is_an_ood_experiment,      # ... choosing the ood... 
                                                     load_adversarial_set=True,     # ... for loading only the adversarial set... 
                                                     num_advx=num_adv_examples)     # ... with 'num_advx' examples... 

    test_subset_loader = torch.utils.data.DataLoader(test_subset_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_subset_loader_during_attack = torch.utils.data.DataLoader(test_subset_set, batch_size=batch_size, shuffle=False, num_workers=16)

    logger.debug("-------------------------")
    logger.debug("----- Clean Results -----")
    logger.debug("-------------------------\n")

    # If the results file does not exist, compute the results, otherwise directly load them
    if not os.path.exists(clean_results_path) or re_evaluation_mode:                       
        logger.debug("Evaluating and saving results...")

        if uq_technique == 'deterministic_uq':
            results = eval.evaluate_deterministic(model, test_subset_loader,        # Evaluating the results on the clean set
                                                  device=device, seed=seed)
        else:
            results = eval.evaluate_bayesian(model, test_subset_loader,             # Evaluating the results on the clean set
                                        mc_sample_size=mc_samples_eval,
                                        device=device, seed=seed)
        utils.utils.my_save(results, clean_results_path)                            # Saving the results
    else:                                                               
        logger.debug("Already evaluated and saved.")
        results = utils.utils.my_load(clean_results_path)                           # Loading the results

    # TODO: Forse sta roba si può incorporare dopo
    # Logging the results
    accuracy = (results['preds'].numpy() == results['ground_truth'].numpy()).mean()
    if uq_technique == 'deterministic_uq':
        conf = results['confidence'].mean().item()
        logger.debug(f"Accuracy: {accuracy:.3f}, Confidence: {conf:.3f}")
    else:
        mi = results['mutual_information'].mean().item()
        logger.debug(f"Accuracy: {accuracy:.3f}, MI: {mi:.3f}")


    '''
        --- Bayesian Model Under Attack ---
        Computing (or loading) the results of the Bayesian model with the selected epsilon perturbation.
    '''

    # We compute the adverarial example generation / valutation only if the epsilon is not 0; otherwise we skip.
    if epsilon != 0:
        logger.debug("----------------")
        logger.debug(attack_loss)
        
        file_count = len(os.listdir(adv_examples_path))
        
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


        # If some adversarial example is missing, compute the remaining ones
        if file_count < num_adv_examples:
            logger.debug("Generating adversarial examples...")
            
            # Set-up phase
            utils.utils.set_all_seed(seed)      # Setting up the seed
            fname_to_target = {}                # Dictionary associating file name with target
            k = 0                               # Sample's index

            # Generating iteratively the adversarial examples from the selected test set
            for batch_i, (x, y) in enumerate(test_subset_loader_during_attack):

                # Sending the data to device
                x, y = x.to(device), y.to(device)                       

                # Computing the adversarial examples for the current batch
                adv_examples = attack.run(x=x, y=y, iterations=num_attack_iterations)
                logger.debug(f"Attack Took {attack.elapsed_time:.3f} seconds ({attack.elapsed_time/num_attack_iterations:.3f} per iter)")

                # Saving each adversarial example on the batch
                for i in range(adv_examples.shape[0]):
                    fname = f"{str(k).zfill(10)}.png"                           # Setting the adversarial image filename
                    file_path = os.path.join(adv_examples_path, fname)          # Creating the path for the adversarial image
                    fname_to_target[fname] = y[i].item()                        # Adding the correspondence to the dictionary
                    torchvision.utils.save_image(adv_examples[i], file_path)    # Saving the adversarial example
                    k += 1                                                      # Incrementing the sample's index

            # Dumping the filename dictionary
            with open(utils.utils.join(adv_examples_path, 'fname_to_target.json'), 'w') as f:   # TODO: Review logic of name dict
                json.dump(fname_to_target, f)
        else:
            logger.debug("Adversarial examples already generated.")
        
        # Loading the adversarial set from the precomputed examples folder
        adv_test_subset_dataset = utils.utils.AdversarialDataset(adv_examples_path)
        adv_test_subset_loader = torch.utils.data.DataLoader(adv_test_subset_dataset, 
                                                        batch_size=batch_size, 
                                                        shuffle=False, 
                                                        num_workers=2)
        

        
        
        # Evaluating the Bayesian model on the adversarial dataset
        if not os.path.exists(adv_results_path) or re_evaluation_mode:        
            logger.debug("Computing evaluation...")
            if uq_technique == 'deterministic_uq':
                adv_results = eval.evaluate_deterministic(model, adv_test_subset_loader,     # Evaluating the results on the clean set
                                                    device=device, seed=seed)
            else:
                adv_results = eval.evaluate_bayesian(model, adv_test_subset_loader, mc_sample_size=mc_samples_eval, seed=seed, device=device)
            utils.utils.my_save(adv_results, adv_results_path)
        else:
            logger.debug("Already evaluated and saved.")
            adv_results = utils.utils.my_load(adv_results_path)                     # Loading the results
        
        # Logging the results
        accuracy = (adv_results['preds'].numpy() == adv_results['ground_truth'].numpy()).mean()
        # mi = adv_results['mutual_information'].mean().item()
        # logger.debug(f"Accuracy: {accuracy:.3f}, MI: {mi:.3f}")

        if uq_technique == 'deterministic_uq':
            conf = adv_results['confidence'].mean().item()
            logger.debug(f"Accuracy: {accuracy:.3f}, Confidence: {conf:.3f}")
        else:
            mi = adv_results['mutual_information'].mean().item()
            logger.debug(f"Accuracy: {accuracy:.3f}, MI: {mi:.3f}")

        print("done!")


'''
    Function for running the main with a given range of epsilons (security evaluation curve)
'''
def main_seceval(parser):

    # Adjusting the parser to add the seceval arguments
    kwargs = {key: value for (key, value) in parser.parse_args()._get_kwargs()}
    parser = add_seceval_parsing(parser)
    args = parser.parse_args()

    # Running the main for each epsilon on the seceval curve
    epsilon_step = (args.epsilon_max - args.epsilon_min) / args.num_epsilon_steps
    epsilon_list = np.arange(start=args.epsilon_min, stop=args.epsilon_max, step=epsilon_step)[::-1]
    for epsilon_k in epsilon_list:
        kwargs['epsilon'] = epsilon_k
        main(**kwargs, kwargs=kwargs)


'''
    Function for running the main with a given epsilon
'''
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

    main_seceval(parser)

