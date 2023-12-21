import utils.utils
from models.utils import load_model
from attacks.bayesian_attacks import StabilizingAttack, AutoTargetAttack
from attacks.loss_functions import UncertaintyDivergenceLoss
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from models.resnet import resnet18
import json
import os
import pickle
from tqdm import tqdm

# --------------------
def val_set_evaluation(model, validation_dataloader, dict_metrics):
        for batch_i, (x, y) in enumerate(validation_dataloader):
            # Sending the data to device
            x, y = x.to(device), y.to(device)                  
            
            # Computing the adv example
            x_adv = attack.run(x=x, y=y, iterations=parameters["attack_iterations"])

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

                clean_out = model(x, mc_sample_size=parameters["mc_samples_attack"])
                adv_out = model(x_adv, mc_sample_size=parameters["mc_samples_attack"])

                # Computing the loss
                loss = loss_adv_training(clean_output=clean_out, adv_output=adv_out, target=y)

                dict_metrics["model_loss_path"].append(loss.item())

                acc = (((F.softmax(clean_out, dim = 1).argmax(dim=1) == y).sum() / parameters["batch_size"]) * 100).item()
                # print(acc)
                dict_metrics["accuracy_path"].append(acc)

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

for b in [4, 7, 8, 10, 15]:


    parameters = {
        "cuda" : 0,

        # DATASET AND MODEL SETTING
        "dataset" : 'cifar10',
        "backbone" : 'resnet18',
        "uq_technique" : 'embedded_dropout',
        "dropout_rate" : 0.1,
    
        # ATTACK SETTINGS
        "epsilon" : 8/255,
        "attack_update_strategy":'pgd',
        "step_size":5e-3,
        "mc_samples_attack" : 5,
        "attack_iterations": 25,

        # TRAIN SETTINGS
        "batch_size" : 256,           # USING 2000
        "epochs" : 10,
        "lr" : 1e-4,                 # 5e-4 usato from scratch, 1e-6 / 1e-7 per finetuning
        "alpha": 1,
        "beta" : b    # OCCHIO che usavo sempre 2 e 8
    }

    print(json.dumps(parameters, indent=4))    
    
    OUTPUT_FOLDER = f"scode_exp/batch_{parameters['batch_size']}/epoch_{parameters['epochs']}\
                                /lr_{parameters['lr']}/alpha{parameters['alpha']}_beta{parameters['beta']}/"

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


    with open(f'{OUTPUT_FOLDER}EXPERIMENT---parameters.txt', 'w') as file:
        file.write(json.dumps(parameters, indent=4))

    device = utils.utils.get_device(parameters["cuda"]) if torch.cuda.is_available() else 'cpu'


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


    model = load_model(parameters["backbone"],
                    parameters["uq_technique"],                              # Loading the model
                    parameters["dataset"],
                    transform=utils.utils.get_normalizer(parameters["dataset"]),
                    dropout_rate=parameters["dropout_rate"],
                    full_bayesian=True,
                    device=device)
    model.to(device)  
    model.train()


    train_set, validation_set, test_set = utils.utils.get_dataset_splits(dataset=parameters["dataset"],
                                                                        set_normalization=False,
                                                                        ood=False,
                                                                        load_adversarial_set=False)

    train_subset_loader_during_attack = torch.utils.data.DataLoader(train_set, batch_size=parameters["batch_size"], shuffle=False, num_workers=16)

    validation_dataloader = torch.utils.data.DataLoader(validation_set, batch_size=parameters["batch_size"], shuffle=False, num_workers=16)


    optimizer = torch.optim.RMSprop(model.parameters(),lr=parameters["lr"])
    loss_adv_training = UncertaintyDivergenceLoss(alpha=parameters["alpha"], beta=parameters["beta"])

    # TODO: Provare un AutoTarget Attack, e cercare (per quanto possibile)
    #       di farlo convergere (i.e., avere UN MINIMO di robustezza in più)
    attack = StabilizingAttack(mc_sample_size_during_attack=parameters["mc_samples_attack"],
                            model=model,
                            device=device,
                            epsilon=parameters["epsilon"],
                            update_strategy=parameters["attack_update_strategy"],
                            step_size=parameters["step_size"])


    # first_batch = next(iter(train_subset_loader_during_attack))


    for e in tqdm(range(parameters["epochs"]), desc="EPOCHS"):
        # Generating iteratively the adversarial examples from the selected test set
        for batch_i, (x, y) in enumerate(tqdm(train_subset_loader_during_attack, desc="BATCHES")):
        #for batch_i, (x, y) in enumerate([first_batch] * 50):

            # Sending the data to device
            x, y = x.to(device), y.to(device)                  
            
            # Computing the adv example
            x_adv = attack.run(x=x, y=y, iterations=parameters["attack_iterations"])
            #optimizer.zero_grad()
            
            # Getting the loss path
            loss_path = attack.loss_fn.loss_path['CE']
            attack.loss_fn.loss_path['CE'] = []

            # Populating the loss paths
            training_metrics["attack_loss_start_path"].append(loss_path[0])
            training_metrics["attack_loss_end_path"].append(loss_path[-1])
            training_metrics["attack_gap_path"].append(loss_path[0] - loss_path[-1])
            # print(f"Attack efficacy: start {loss_path[0]} end {loss_path[-1]}; delta {loss_path[0]-loss_path[-1]}")

            # Feeding the model with the clean and adv example
            # for e_inner in range(parameters["attack_iterations"]):
            optimizer.zero_grad()

            clean_out = model(x, mc_sample_size=parameters["mc_samples_attack"])
            adv_out = model(x_adv, mc_sample_size=parameters["mc_samples_attack"])

            # Computing the loss
            loss = loss_adv_training(clean_output=clean_out, adv_output=adv_out, target=y)
            # print(f"{loss.item()=}")

            # Backpropagation
            loss.backward()
            optimizer.step()

            training_metrics["model_loss_path"].append(loss.item())

            acc = (((F.softmax(clean_out, dim = 1).argmax(dim=1) == y).sum() / parameters["batch_size"]) * 100).item()
            # print(acc)
            training_metrics["accuracy_path"].append(acc)
            
            if batch_i%25 == 0:
                # print(f"{batch_i=}, {loss.item()=}")

                loss_adv_training.plot_loss(fig_path=f'{OUTPUT_FOLDER}EXPERIMENT---model_loss.png')
                print_metrics(training_metrics, fig_path = f'{OUTPUT_FOLDER}EXPERIMENT---aggregate_losses.png')
                torch.save(model.state_dict(), f'{OUTPUT_FOLDER}EXPERIMENT---model-weights.pt')

                with open(f'{OUTPUT_FOLDER}EXPERIMENT---metrics_dict.pkl', "wb") as f:
                    pickle.dump(training_metrics, f)


    print_metrics(training_metrics, fig_path = f'{OUTPUT_FOLDER}EXPERIMENT---aggregate_losses.png')
    torch.save(model.state_dict(), f'{OUTPUT_FOLDER}EXPERIMENT---model-weights.pt')
    
    with open(f'{OUTPUT_FOLDER}EXPERIMENT---metrics_dict.pkl', "wb") as f:
        pickle.dump(training_metrics, f)


        # if batch_i%20 == 0:
        #     val_set_evaluation(model ,validation_dataloader, validation_metrics)
        #     print_metrics(validation_metrics, fig_path = f'{OUTPUT_FOLDER}EXPERIMENT-alpha{parameters["alpha"]}-beta{parameters["beta"]}---aggregate_losses_VALIDATION.png')
