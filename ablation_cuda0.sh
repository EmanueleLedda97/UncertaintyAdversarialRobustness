#!bin/bash

python main_attack.py -dataset imagenet -robustness_level semi_robust \
-robust_model salman2020R18 -num_adv_examples 5000 -cuda 0

python main_attack.py -dataset imagenet -robustness_level semi_robust \
-robust_model wong2020 -num_adv_examples 5000 -cuda 0

python main_attack.py -dataset imagenet -robustness_level semi_robust \
-robust_model engstrom2019imgnet -num_adv_examples 5000 -cuda 0

python main_attack.py -dataset imagenet -robustness_level semi_robust \
-robust_model salman2020R50 -num_adv_examples 5000 -cuda 0
