
# Lists of all the supported Backbones
SUPPORTED_RESNETS = ["resnet18", "resnet34", "resnet50", "resnet_fcn"]
SUPPORTED_VGGS = []
SUPPORTED_BACKBONES = SUPPORTED_RESNETS + SUPPORTED_VGGS

# Lists of supported data sets and UQ methods
SUPPORTED_DATASETS = ['cifar10', 'cifar100']
SUPPORTE_UQ_METHODS = ['embedded_dropout', 'injected_dropout', 'deep_ensemble', 'deterministic_uq']

# Lists of supported experiments
ROBUSTNESS_LEVELS = ['naive_robust', 'semi_robust', 'full_robust']
EXPERIMENT_CATEGORIES = ['classification_id', 'classification_ood', 'semantic_segmentation']

# Lists of supported attacks
SUPPORTED_UNDERCONFIDENCE_ATTACKS = ['MaxVar']
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
EPS_BASE = 0.031                    # 8/255
BASE_EPSILON = 0.031                # Just a refactoring of the upper constant
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