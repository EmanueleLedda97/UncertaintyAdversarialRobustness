# Lists of all the supported Backbones
SUPPORTED_RESNETS = ["resnet18", "resnet34", "resnet50", "resnet_fcn", 'robust_resnet',
                     'ConvNeXt-L', 'ConvNeXt-B', 'Swin-B', 'Swin-L',
                     'RaWideResNet-70-16',
                     'WideResNet-70-16',
                     'WideResNet-28-10', 'WideResNet-34-10']

SUPPORTED_VGGS = []
SUPPORTED_BACKBONES = SUPPORTED_RESNETS + SUPPORTED_VGGS

# Lists of supported data sets and UQ methods
SUPPORTED_DATASETS = ['cifar10', 'cifar100', "imagenet"]
SUPPORTE_UQ_METHODS = ["None", 'embedded_dropout', 'injected_dropout', 'deep_ensemble', 'deterministic_uq']

# Lists of supported experiments
ROBUSTNESS_LEVELS = ['naive_robust', 'semi_robust', 'full_robust']
EXPERIMENT_CATEGORIES = ['classification_id', 'classification_ood', 'semantic_segmentation']

# Lists of supported attacks
SUPPORTED_UNDERCONFIDENCE_ATTACKS = ['MaxVar', 'Shake']
SUPPORTED_OVERCONFIDENCE_ATTACKS = ['MinVar', 'AutoTarget', 'Stab', 'Centroid', 'UST']
SUPPORTED_ATTACKS = SUPPORTED_UNDERCONFIDENCE_ATTACKS + SUPPORTED_OVERCONFIDENCE_ATTACKS
SUPPORTED_UPDATE_STRATEGIES = ['pgd', 'fgsm']

SUPPORTED_NORMS = ['Linf', 'L2']

# List of supported dropout rates and ensemble sizes
SUPPORTED_DROPOUT_RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
MAXIMUM_ENSEMBLE_SIZE = 5

# Dictionary containing the number of classes for each dataset
NUBMER_OF_CLASSES_PER_DATASET = {
    'cifar10': 10,
    'cifar100': 100
}

# The last hyperparameters chosen during the optimization process (investigate on these)
LAST_HYPERPARAMETERS_CHOICES = (2e-3, 150)
BATCH_FOR_EACH_BACKBONE = {
    'resnet18': (150, 400),
    'resnet34': (64, 128),
    'resnet50': (8, 32)
}

LAST_HYPERPARAMETERS_CHOICES_DUQ = (1e-3, 100)
BATCH_FOR_EACH_BACKBONE_DUQ = {
    'resnet18': (1000, 1000),
    'resnet34': (1000, 1000),
    'resnet50': (1000, 1000)
}

# Seed used for data reproducibility. Should always be 42
DATA_SEED = 42
SUPPORTED_CUDAS = [0, 1]

# Attack constants
ROOT = 'experiments_correct'
EPS_BASE = 0.031  # 8/255
BASE_EPSILON = 0.031  # Just a refactoring of the upper constant
OPTIM_ATK_TYPE = ('fgsm', 'pgd')

SEL_LOSS_TERMS = {'pred': (1, 0),
                  'unc': (0, 1),
                  'both': (1, 1)}

# Dictionary containing all the normalization vectors for each data set
NORMALIZATION_DICT = {
    'cifar10': ([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]),
    'imagenet': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
}

DEFAULT_DROPOUT_RATE = 0.3



# The order change the display of plots in plot_results.py
# Architecture order aka how the file are saved in folder
# CIFAR10_ROBUST_MODELS = ['addepalli2022', 'addepalli2022_towards', 'sehwag2021', # RESNET-18
#                          'engstrom2019', # RESNET-50
#                          'sehwag2021Proxy_ResNest152', # RESNET-152
#                          'Cui2023Decoupled_WRN_28_10', 'gowal2021Improving_28_10',
#                          'Wang2023Better_WRN_28_10', 'Xu2023Exploring_WRN_28_10', # WRN-28-10
#                          'Gowal2021Improving_70_16_ddpm_100m', 'kang2021Stable',
#                          'pang2022Robustness_WRN70_16', 'Rebuffi2021Fixing_70_16_cutmix_extra',
#                          'Wang2023Better_WRN_70_16', # WRN-70-10
#                          'Peng2023Robust' #RaWRN-70-16
#                          ]

# cifar_latex_command_id = ["\\addepallieffID","\\addepallitowID","\sehwagrobustID","\engstromID","\sehwagrobustIDdue","\cuidecoupledID","\gowalimprovingID",
# "\wangbetterID","\\xuexploringID","\gowalimprovingIDdue","\kangstableID","\pangrobustnessID","\\rebuffifixingID", "\wangbetterIDdue", "\pengrobustID"]


# PAPER CITATION ORDER
CIFAR10_ROBUST_MODELS = ['engstrom2019', 'addepalli2022', 'gowal2021Improving_28_10', 'Gowal2021Improving_70_16_ddpm_100m',
                         'Wang2023Better_WRN_28_10', 'Wang2023Better_WRN_70_16', 'sehwag2021', 'sehwag2021Proxy_ResNest152',
                         'Rebuffi2021Fixing_70_16_cutmix_extra', 'kang2021Stable', 'Peng2023Robust', 'addepalli2022_towards',
                         'Cui2023Decoupled_WRN_28_10', 'Xu2023Exploring_WRN_28_10', 'pang2022Robustness_WRN70_16']

cifar_latex_command_id = ["\engstromID","\\addepallieffID","\gowalimprovingID","\gowalimprovingIDdue",
"\wangbetterID","\wangbetterIDdue","\sehwagrobustID","\sehwagrobustIDdue",
"\\rebuffifixingID","\kangstableID","\pengrobustID","\\addepallitowID",
"\cuidecoupledID","\\xuexploringID","\pangrobustnessID"]


# CIFAR10_DICT_MODELS_TO_ID = dict((name, cifar_latex_command_id[idx]) for idx, name in enumerate(CIFAR10_ROBUST_MODELS))
CIFAR10_DICT_MODELS_TO_ID = {
'engstrom2019': {"paper_id":'\\engstromID',
                 "paper_ref":"\\engstrom",
                 "rb_rank":"15",
                 "clean_acc": "87.03\%",
                 "rob_acc": "49.25\%"},
 'addepalli2022': {"paper_id":'\\addepallieffID',
                   "paper_ref":"\\addepallieff",
                   "rb_rank":"13",
                   "clean_acc": "85.71\%",
                  "rob_acc": "52.48\%",
                   },
 'gowal2021Improving_28_10': {"paper_id":'\\gowalimprovingID',
                              "paper_ref":"\\gowalimproving",
                              "rb_rank":"9",
                              "clean_acc":"87.50\%",
                              "rob_acc":"63.38\%",
                              },
 'Gowal2021Improving_70_16_ddpm_100m': {"paper_id":'\\gowalimprovingIDdue',
                                        "paper_ref":"\\gowalimproving",
                                        "rb_rank":"6",
                                        "clean_acc":"88.74\%",
                                        "rob_acc":"66.10\%",
                                        },
 'Wang2023Better_WRN_28_10': {"paper_id":'\\wangbetterID',
                              "paper_ref":"\\wangbetter",
                              "rb_rank":"4",
                              "clean_acc":"92.44\%",
                              "rob_acc":"67.31\%",
                              },
 'Wang2023Better_WRN_70_16': {"paper_id":'\\wangbetterIDdue',
                              "paper_ref":"\\wangbetterID",
                              "rb_rank":"2",
                              "clean_acc":"70.69\%",
                              "rob_acc":"70.69\%",
                              },
 'sehwag2021': {"paper_id":'\\sehwagrobustID',
                "paper_ref":"\\sehwagrobust",
                "rb_rank":"12",
                "clean_acc":"84.59\%",
                "rob_acc":"55.54\%",
                },
 'sehwag2021Proxy_ResNest152': {"paper_id":'\\sehwagrobustIDdue',
                                "paper_ref":"\\sehwagrobust",
                                "rb_rank":"11",
                                "clean_acc":"87.30\%",
                                "rob_acc":"62.79\%",
                                },
 'Rebuffi2021Fixing_70_16_cutmix_extra': {"paper_id":'\\rebuffifixingID',
                                          "paper_ref":"\\rebuffifixing",
                                          "rb_rank":"5",
                                          "clean_acc":"92.23\%",
                                          "rob_acc":"66.56\%",
                                          },
 'kang2021Stable': {"paper_id":'\\kangstableID',
                    "paper_ref":"\\kangstable",
                    "rb_rank":"7",
                    "clean_acc":"93.73\%",
                    "rob_acc":"64.20\%",
                    },
 'Peng2023Robust': {"paper_id":'\\pengrobustID',
                    "paper_ref":"\\pengrobust",
                    "rb_rank":"1",
                    "clean_acc":"93.27\%",
                    "rob_acc":"71.07\%",
                    },
 'addepalli2022_towards': {"paper_id":'\\addepallitowID',
                           "paper_ref":"\\addepallitow",
                           "rb_rank":"14",
                           "clean_acc":"80.24\%",
                           "rob_acc":"51.06\%",
                           },
 'Cui2023Decoupled_WRN_28_10': {"paper_id":'\\cuidecoupledID',
                                "paper_ref":"\\cuidecoupled",
                                "rb_rank":"3",
                                "clean_acc":"92.16\%",
                                "rob_acc":"67.73\%",
                                },
 'Xu2023Exploring_WRN_28_10': {"paper_id":'\\xuexploringID',
                               "paper_ref":"\\xuexploring",
                               "rb_rank":"8",
                               "clean_acc":"93.69\%",
                               "rob_acc":"63.89\%",
                               },
 'pang2022Robustness_WRN70_16': {"paper_id":'\\pangrobustnessID',
                                 "paper_ref":"\\pangrobustness",
                                 "rb_rank":"10",
                                 "clean_acc":"89.01\%",
                                 "rob_acc":"63.35\%",
                                 }
}


# CREA UGUALE PER IMAGENET
# Architecture order aka how the file are saved in folder
# IMAGENET_ROBUST_MODELS = ['salman2020R18', # RESNET-18
#                           'engstrom2019imgnet', 'salman2020R50', 'wong2020', # RESNET-50
#                           'Liu2023swinB', 'Liu2023swinL', # SWIN
#                           'Liu2023convNextB', 'Liu2023convNextL', # CONVNEXT
#                           ]
# imagenet_latex_command_id = ["\salmanID","\engstromimagenetID","\salmanIDdue","\wongfastID",
# "\liuswinb","\liuswinl","\liuconvnb","\liuconvnl"]

# PAPER CITATION ORDER
IMAGENET_ROBUST_MODELS = ['engstrom2019imgnet', 'salman2020R18', 'salman2020R50', 'wong2020',
                          'Liu2023swinB', 'Liu2023swinL', 'Liu2023convNextB', 'Liu2023convNextL']

imagenet_latex_command_id = ["\engstromimagenetID", "\salmanID","\salmanIDdue","\wongfastID",
"\liuswinb","\liuswinl","\liuconvnb","\liuconvnl"]

# IMAGENET_DICT_MODELS_TO_ID = dict((name, imagenet_latex_command_id[idx]) for idx, name in enumerate(IMAGENET_ROBUST_MODELS))

IMAGENET_DICT_MODELS_TO_ID ={
'engstrom2019imgnet': {"paper_id":'\\engstromimagenetID',
                       "paper_ref":"\\engstromimagenet",
                    "rb_rank":"6",
                    "clean_acc":"62.56\%",
                    "rob_acc":"29.22\%"
                       },
 'salman2020R18': {"paper_id":'\\salmanID',
                   "paper_ref":"\\salman",
                   "rb_rank":"8",
                   "clean_acc":"52.92\%",
                   "rob_acc":"25.32\%"
                   },
 'salman2020R50': {"paper_id":'\\salmanIDdue',
                   "paper_ref":"\\salman",
                   "rb_rank":"5",
                   "clean_acc":"64.02\%",
                   "rob_acc":"34.96\%"
                   },
 'wong2020': {"paper_id":'\\wongfastID',
              "paper_ref":"\\wongfast",
              "rb_rank":"7",
              "clean_acc":"55.62\%",
              "rob_acc":"26.24\%"
              },
 'Liu2023swinB': {"paper_id":'\\liuswinb',
                  "paper_ref":"\\liu",
                  "rb_rank":"3",
                  "clean_acc":"76.16\%",
                  "rob_acc":"56.16\%"
                  },
 'Liu2023swinL': {"paper_id":'\\liuswinl',
                  "paper_ref":"\\liu",
                  "rb_rank":"1",
                  "clean_acc":"78.92\%",
                  "rob_acc":"59.56\%"
                  },
 'Liu2023convNextB': {"paper_id":'\\liuconvnb',
                      "paper_ref":"\\liu",
                      "rb_rank":"4",
                      "clean_acc":"76.02\%",
                      "rob_acc":"55.82\%"
                      },
 'Liu2023convNextL': {"paper_id":'\\liuconvnl',
                      "paper_ref":"\\liu",
                      "rb_rank":"2",
                      "clean_acc":"78.02\%",
                      "rob_acc":"58.48\%"
                      }
 }



CIFAR10_NAIVE_MODELS = ['resnet18', 'resnet34', 'resnet50']

IMAGENET_NAIVE_MODELS = ['resnet18', 'resnet50', 'ConvNeXt-L', 'ConvNeXt-B', 'Swin-B']


L2_ROBUST_MODELS = ['sehwag2021', 'engstrom2019', 'augustin2020']
LINF_ROBUST_MODELS = ['addepalli2022', 'addepalli2022_towards', 'sehwag2021', # RESNET-18
                         'engstrom2019', # RESNET-50
                         'sehwag2021Proxy_ResNest152', # RESNET-152
                         'Cui2023Decoupled_WRN_28_10', 'gowal2021Improving_28_10',
                         'Wang2023Better_WRN_28_10', 'Xu2023Exploring_WRN_28_10', # WRN-28-10
                         'Gowal2021Improving_70_16_ddpm_100m', 'kang2021Stable',
                         'pang2022Robustness_WRN70_16', 'Rebuffi2021Fixing_70_16_cutmix_extra',
                         'Wang2023Better_WRN_70_16', # WRN-70-10
                         'Peng2023Robust',  # RaWRN-70-16

                      'salman2020R18', # RESNET-18
                          'engstrom2019imgnet', 'salman2020R50', 'wong2020', # RESNET-50
                          'Liu2023convNextL', 'Liu2023swinB', # SWIN
                          'Liu2023convNextB', 'Liu2023swinL' # CONVNEXT
                      ]

SUPPORTED_ROBUST_MODEL = CIFAR10_ROBUST_MODELS + IMAGENET_ROBUST_MODELS


cifar10_model_dict = dict(
    addepalli2022={
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
    ######## New Entries #############
    Peng2023Robust={
        'name': 'Peng2023Robust',
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf',
        'resnet_type': 'RaWideResNet-70-16'
    },
    Wang2023Better_WRN_70_16={
        'name': 'Wang2023Better_WRN-70-16',
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf',
        'resnet_type': 'WideResNet-70-16'
    },
    Cui2023Decoupled_WRN_28_10={
        'name': 'Cui2023Decoupled_WRN-28-10',
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf',
        'resnet_type': 'WideResNet-28-10'
    },
    Wang2023Better_WRN_28_10={
        'name': 'Wang2023Better_WRN-28-10',
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf',
        'resnet_type': 'WideResNet-28-10'
    },
    Rebuffi2021Fixing_70_16_cutmix_extra={
        'name': 'Rebuffi2021Fixing_70_16_cutmix_extra',
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf',
        'resnet_type': 'WideResNet-70-16'
    },
    Gowal2021Improving_70_16_ddpm_100m={
        'name': 'Gowal2021Improving_70_16_ddpm_100m',
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf',
        'resnet_type': 'WideResNet-70-16'
    },
    Xu2023Exploring_WRN_28_10={
        'name': 'Xu2023Exploring_WRN-28-10',
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf',
        'resnet_type': 'WideResNet-28-10'
    },
    Sehwag2021Proxy={
        'name': 'Sehwag2021Proxy',
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf',
        'resnet_type': 'WideResNet-34-10'
    },
)

imagenet_model_dict = dict(
    salman2020R18={
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