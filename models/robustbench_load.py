from functools import partial

from torch import nn

from robustbench import load_model


def load_robustbench_model(name: str, dataset: str, threat_model: str, device: str = "cpu", **kwargs) -> nn.Module:
    model = load_model(model_name=name, dataset=dataset, threat_model=threat_model)
    return model


# ---------------- CIFAR-10 ----------------------

cifar10_model_dict = dict(addepalli2022={
    'name': 'Addepalli2022Efficient_RN18',  # ResNet-18
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf',
    'resnet_type': 'resnet18'
    },
    addepalli2022_towards={
    'name': 'Addepalli2021Towards_RN18',  # ResNet-18
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf',
    'resnet_type': 'resnet18'
    },
    sehwag2021={
    'name': 'Sehwag2021Proxy_R18',  # ResNet-18
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf',  # Available [Linf, L2]
    'resnet_type': 'resnet18'
    },
    engstrom2019={
    'name': 'Engstrom2019Robustness',  # RESNET50
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf',  # training threat model. Available [Linf, L2]
    'resnet_type': 'resnet50'
    },
    augustin2020={
    'name': 'Augustin2020Adversarial',  # RESNET50
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'L2',  # training threat model
    'resnet_type': 'resnet50'
    },
    sehwag2021Proxy_ResNest152={
        'name': 'Sehwag2021Proxy_ResNest152',  # RESNET50
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf',  # training threat model
        'resnet_type': 'resnet152'
        },
    pang2022Robustness_WRN70_16={
        'name': 'Pang2022Robustness_WRN70_16',  # RESNET50
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf',  # training threat model
        'resnet_type': 'WideResNet-70-16'
    },
    gowal2021Improving_28_10={
        'name': 'Gowal2021Improving_28_10_ddpm_100m',  # RESNET50
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf',  # training threat model
        'resnet_type': 'WideResNet-28-10'
    },
    kang2021Stable={
        'name': 'Kang2021Stable',  # RESNET50
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf',  # training threat model
        'resnet_type': 'WideResNet-70-16'
    },
)


_local_cifar_models = {
    'addepalli2022_towards': partial(load_robustbench_model, **cifar10_model_dict["addepalli2022_towards"]),
    'addepalli2022': partial(load_robustbench_model, **cifar10_model_dict["addepalli2022"]),
    'sehwag2021': partial(load_robustbench_model, **cifar10_model_dict["sehwag2021"]),
    'augustin2020': partial(load_robustbench_model, **cifar10_model_dict["augustin2020"]),
    'engstrom2019': partial(load_robustbench_model, **cifar10_model_dict["engstrom2019"]),
    'sehwag2021Proxy_ResNest152': partial(load_robustbench_model, **cifar10_model_dict["sehwag2021Proxy_ResNest152"]),
    'pang2022Robustness_WRN70_16': partial(load_robustbench_model, **cifar10_model_dict["pang2022Robustness_WRN70_16"]),
    'gowal2021Improving_28_10': partial(load_robustbench_model, **cifar10_model_dict["gowal2021Improving_28_10"]),
    'kang2021Stable': partial(load_robustbench_model, **cifar10_model_dict["kang2021Stable"]),

}


def get_local_cifar_model(name: str, dataset: str) -> nn.Module:
    return _local_cifar_models[name]()


# -------------------- IMAGENET -----------------------------
# def get_resnet18():
#     model = resnet18_normalized(weights= ResNet18_Weights.DEFAULT, normalize=([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]))
#     return model Accuracy: 0.455, MI: 3.086

imagenet_model_dict = dict(salman2020R18={
    'name': 'Salman2020Do_R18',  # ResNet-18
    'source': 'robustbench',
    'dataset': 'imagenet',
    'threat_model': 'Linf',
    'resnet_type': 'resnet18'
    },
    wong2020={
    'name': 'Wong2020Fast',  # ResNet-50
    'source': 'robustbench',
    'dataset': 'imagenet',
    'threat_model': 'Linf',
    'resnet_type': 'resnet50'
    },
    engstrom2019imgnet={
    'name': 'Engstrom2019Robustness',  # ResNet-50
    'source': 'robustbench',
    'dataset': 'imagenet',
    'threat_model': 'Linf',
    'resnet_type': 'resnet50'
    },
    salman2020R50={
    'name': 'Salman2020Do_R50',  # ResNet-50
    'source': 'robustbench',
    'dataset': 'imagenet',
    'threat_model': 'Linf',
    'resnet_type': 'resnet50'
    },

    Liu2023convNextL={
        'name': 'Liu2023Comprehensive_ConvNeXt-L',
        'source': 'robustbench',
        'dataset': 'imagenet',
        'threat_model': 'Linf',
        'resnet_type': 'ConvNeXt-L'
        },
    Liu2023swinB={
            'name': 'Liu2023Comprehensive_Swin-B',
            'source': 'robustbench',
            'dataset': 'imagenet',
            'threat_model': 'Linf',
            'resnet_type': 'Swin-B'
        },
    Liu2023convNextB={
        'name': 'Liu2023Comprehensive_ConvNeXt-B',
        'source': 'robustbench',
        'dataset': 'imagenet',
        'threat_model': 'Linf',
        'resnet_type': 'ConvNeXt-B'
    },
    Liu2023swinL={
        'name': 'Liu2023Comprehensive_Swin-L',
        'source': 'robustbench',
        'dataset': 'imagenet',
        'threat_model': 'Linf',
        'resnet_type': 'Swin-L'
    },
)

_local_imagenet_models = {
    'salman2020R18': partial(load_robustbench_model, **imagenet_model_dict["salman2020R18"]),
    'wong2020': partial(load_robustbench_model, **imagenet_model_dict["wong2020"]),
    'engstrom2019imgnet': partial(load_robustbench_model, **imagenet_model_dict["engstrom2019imgnet"]),
    'salman2020R50': partial(load_robustbench_model, **imagenet_model_dict["salman2020R50"]),

    'Liu2023convNextL': partial(load_robustbench_model, **imagenet_model_dict["Liu2023convNextL"]),
    'Liu2023swinB': partial(load_robustbench_model, **imagenet_model_dict["Liu2023swinB"]),
    'Liu2023convNextB': partial(load_robustbench_model, **imagenet_model_dict["Liu2023convNextB"]),
    'Liu2023swinL': partial(load_robustbench_model, **imagenet_model_dict["Liu2023swinL"]),

}


def get_local_model(name: str, dataset: str, device: str = "cpu") -> nn.Module:
    print(f"Loading {name}")
    if dataset == 'cifar10':
        return _local_cifar_models[name](device=device)
    elif dataset == 'imagenet':
        return _local_imagenet_models[name](device=device)
