#!bin/bash

#python main_attack.py -dataset imagenet -robustness_level semi_robust \
#-robust_model Liu2023convNextL -num_adv_examples 5000 -cuda 1 \
#-batch_size 16 -epsilon 0.0157
#
#python main_attack.py -dataset imagenet -robustness_level semi_robust \
#-robust_model Liu2023convNextB -num_adv_examples 5000 -cuda 1 \
#-batch_size 16 -epsilon 0.0157

#python main_attack.py -dataset imagenet -backbone resnet18 -robustness_level naive_robust \
# -uq_technique None -num_adv_examples 5000 -cuda 1 \
#-batch_size 64 -epsilon 0.0157
#
#python main_attack.py -dataset imagenet -backbone resnet50 -robustness_level naive_robust \
# -uq_technique None -num_adv_examples 5000 -cuda 1 \
#-batch_size 64 -epsilon 0.0157 -re_evaluation_mode True


python main_attack.py -dataset imagenet -robustness_level semi_robust \
-robust_model Liu2023swinL -num_adv_examples 5000 -cuda 1 \
-batch_size 16 -epsilon 0.0314 -re_evaluation_mode True