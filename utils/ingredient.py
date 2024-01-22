from functools import partial

from torch import nn

from robustbench import load_model




# ---------------- CIFAR-10 ----------------------

addepalli2022 = {
    'name': 'Addepalli2022Efficient_RN18', # ResNet-18
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf'
}

sehwag2021 = {
    'name': 'Sehwag2021Proxy_R18', # ResNet-18
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf' # Available [Linf, L2]
}

engstrom_2019 = {
    'name': 'Engstrom2019Robustness',  # RESNET50
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'L2'  # training threat model. Available [Linf, L2]
}

augustin_2020 = {
    'name': 'Augustin2020Adversarial', # RESNET50
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'L2'  # training threat model
}

def load_robustbench_model(name: str, dataset: str, threat_model: str, device:str="cpu") -> nn.Module:
    model = load_model(model_name=name, dataset=dataset, threat_model=threat_model)
    return model

_local_cifar_models = {
    'addepalli2022': partial(load_robustbench_model, name=addepalli2022['name'], dataset=addepalli2022['dataset'], threat_model=addepalli2022['threat_model']),
    'sehwag2021': partial(load_robustbench_model, name=sehwag2021['name'], dataset=sehwag2021['dataset'], threat_model=sehwag2021['threat_model']),
    'augustin2020': partial(load_robustbench_model, name=augustin_2020['name'], dataset=augustin_2020['dataset'], threat_model=augustin_2020['threat_model']),
    'engstrom2019': partial(load_robustbench_model, name=engstrom_2019['name'], dataset=engstrom_2019['dataset'], threat_model=engstrom_2019['threat_model']),
}

def get_local_cifar_model(name: str, dataset: str) -> nn.Module:
    return _local_cifar_models[name]()



# -------------------- IMAGENET -----------------------------
# def get_resnet18():
#     model = resnet18_normalized(weights= ResNet18_Weights.DEFAULT, normalize=([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]))
#     return model

salman_2020R18 = {
    'name': 'Salman2020Do_R18', # ResNet-18
    'source': 'robustbench',
    'dataset': 'imagenet',
    'threat_model': 'Linf'
}

wong_2020 = {
    'name': 'Wong2020Fast', # ResNet-50
    'source': 'robustbench',
    'dataset': 'imagenet',
    'threat_model': 'Linf'
}

engstrom_2019_imgnet = {
    'name': 'Engstrom2019Robustness', # ResNet-50
    'source': 'robustbench',
    'dataset': 'imagenet',
    'threat_model': 'Linf'
}

Salman2020R50 = {
    'name': 'Salman2020Do_R50', # ResNet-50
    'source': 'robustbench',
    'dataset': 'imagenet',
    'threat_model': 'Linf'
}



_local_imagenet_models = {
    'salman2020R18': partial(load_robustbench_model, name=salman_2020R18['name'], dataset=salman_2020R18['dataset'], threat_model=salman_2020R18['threat_model']),
    'wong2020': partial(load_robustbench_model, name=wong_2020['name'], dataset=wong_2020['dataset'], threat_model=wong_2020['threat_model']),
    'engstrom2019imgnet': partial(load_robustbench_model, name=engstrom_2019_imgnet['name'], dataset=engstrom_2019_imgnet['dataset'], threat_model=engstrom_2019_imgnet['threat_model']),
    'salman2020R50': partial(load_robustbench_model, name=Salman2020R50['name'], dataset=Salman2020R50['dataset'], threat_model=Salman2020R50['threat_model'])
}


def get_local_model(name: str, dataset: str, device:str = "cpu") -> nn.Module:
    print(f"Loading {name}")
    if dataset == 'cifar10':
        return _local_cifar_models[name](device=device)
    elif dataset == 'imagenet':
        return _local_imagenet_models[name](device)