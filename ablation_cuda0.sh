#!bin/bash

# python main_attack.py -dataset imagenet -robustness_level semi_robust \
# -robust_model salman2020R18 -num_adv_examples 5000 -cuda 0 -re_evaluation_mode True \
# -batch_size 64 -epsilon 0.0157

# python main_attack.py -dataset imagenet -robustness_level semi_robust \
# -robust_model wong2020 -num_adv_examples 5000 -cuda 0 -re_evaluation_mode True \
# -batch_size 64 -epsilon 0.0157

# python main_attack.py -dataset imagenet -robustness_level semi_robust \
# -robust_model engstrom2019imgnet -num_adv_examples 5000 -cuda 0 -re_evaluation_mode True \
# -batch_size 64 -epsilon 0.0157

# python main_attack.py -dataset imagenet -robustness_level semi_robust \
# -robust_model salman2020R50 -num_adv_examples 5000 -cuda 0 -re_evaluation_mode True \
# -batch_size 64 -epsilon 0.0157

# 8/255
for EPS in 0.0314 0.0274 0.0235 0.0196 0.0156 0.0118 0.0078 0.0039
do
    # python main_attack.py -dataset cifar10 -robustness_level semi_robust -robust_model addepalli2022 -num_adv_examples 5000 -cuda 0 -re_evaluation_mode True -batch_size 5000 -epsilon $EPS
    # wait

    # python main_attack.py -dataset cifar10 -robustness_level semi_robust -robust_model addepalli2022_towards -num_adv_examples 5000 -cuda 0 -re_evaluation_mode True -batch_size 5000 -epsilon $EPS
    # wait

    # python main_attack.py -dataset cifar10 -robustness_level semi_robust -robust_model sehwag2021 -num_adv_examples 5000 -cuda 0 -re_evaluation_mode True -batch_size 5000 -epsilon $EPS
    # wait

    python main_attack.py -dataset cifar10 -robustness_level semi_robust -robust_model engstrom2019 -num_adv_examples 5000 -cuda 0 -re_evaluation_mode True -batch_size 1000 -epsilon $EPS
    wait
done