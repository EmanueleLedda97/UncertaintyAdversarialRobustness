import utils.utils
from models.utils import load_model
from attacks.bayesian_attacks import StabilizingAttack
from attacks.loss_functions import UncertaintyDivergenceLoss
import torch

dataset = 'cifar10'
backbone = 'resnet18'
uq_technique = 'embedded_dropout'
dropout_rate = 0.3
cuda = 0


device = utils.utils.get_device(cuda) if torch.cuda.is_available() else 'cpu'
model = load_model(backbone,
                   uq_technique,                              # Loading the model
                   dataset,
                   transform=utils.utils.get_normalizer(dataset),
                   dropout_rate=dropout_rate,
                   full_bayesian=True,
                   device=device)

model.to(device)  

epsilon = 16/255
attack_update_strategy='pgd'
step_size=1
mc_samples_attack = 5
batch_size = 64
attack = StabilizingAttack(mc_sample_size_during_attack=mc_samples_attack,
                           model=model,
                           device=device,
                           epsilon=epsilon,
                           update_strategy=attack_update_strategy,
                           step_size=step_size)


train_set, validation_set, test_set = utils.utils.get_dataset_splits(dataset=dataset,
                                                                     set_normalization=False,
                                                                     ood=False,
                                                                     load_adversarial_set=False)

train_subset_loader_during_attack = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=16)

lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_adv_training = UncertaintyDivergenceLoss(alpha=0.0, beta=1.0)

epochs = 100
for e in range(epochs):
    # Generating iteratively the adversarial examples from the selected test set
    for batch_i, (x, y) in enumerate(train_subset_loader_during_attack):

        # Sending the data to device
        x, y = x.to(device), y.to(device)                       

        model.train()
        # NOTE: Ricordiamoci che ci servono i gradienti
        # Computing the adversarial examples for the current batch
        x_adv = attack.run(x=x, y=y, iterations=5)

        clean_out = model(x, mc_sample_size=5)
        adv_out = model(x_adv, mc_sample_size=5)
        loss = loss_adv_training(clean_out, adv_out, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(loss)


    # TODO: Capire come e dove salvare i modelli
    
