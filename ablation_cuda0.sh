#!bin/bash

CUDA=0

      # ROBUST MODELS

#for EPS in 0.0313 0.0157
#do
#    python main_attack.py -dataset imagenet -robustness_level semi_robust -robust_model salman2020R18 -num_adv_examples 10000\
#                          -cuda $CUDA -re_evaluation_mode True -batch_size 400 -epsilon $EPS
#    wait
#    python main_attack.py -dataset imagenet -robustness_level semi_robust -robust_model wong2020 -num_adv_examples 10000\
#                          -cuda $CUDA -re_evaluation_mode True -batch_size 400 -epsilon $EPS
#    wait
#    python main_attack.py -dataset imagenet -robustness_level semi_robust -robust_model engstrom2019imgnet -num_adv_examples 10000\
#                          -cuda $CUDA -re_evaluation_mode True -batch_size 400 -epsilon $EPS
#    wait
#    python main_attack.py -dataset imagenet -robustness_level semi_robust -robust_model salman2020R50 -num_adv_examples 10000\
#                          -cuda $CUDA -re_evaluation_mode True -batch_size 400 -epsilon $EPS
#    wait
#done