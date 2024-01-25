#!bin/bash

python main_attack.py -dataset imagenet -backbone resnet18 -robustness_level naive_robust \
 -uq_technique None -num_adv_examples 5000 -cuda 1

python main_attack.py -dataset imagenet -backbone resnet50 -robustness_level naive_robust \
 -uq_technique None -num_adv_examples 5000 -cuda 1

python main_attack.py -dataset cifar10 -backbone resnet18 -robustness_level naive_robust \
 -uq_technique None -num_adv_examples 5000 -cuda 1

 python main_attack.py -dataset cifar10 -backbone resnet50 -robustness_level naive_robust \
 -uq_technique None -num_adv_examples 5000 -cuda 1

