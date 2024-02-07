#!bin/bash


CUDA=1

for EPS in 0.0313 0.0157
do
#  python main_attack.py -dataset cifar10 -robustness_level semi_robust -robust_model sehwag2021 -num_adv_examples 1000\
#                        -cuda $CUDA -re_evaluation_mode True -batch_size 128 -epsilon $EPS -num_attack_iterations 150\
#                        -step_size 0.002 -attack_loss Shake
#  wait
  python main_attack.py -dataset cifar10 -robustness_level semi_robust -robust_model addepalli2022 -num_adv_examples 1000\
                        -cuda $CUDA -re_evaluation_mode True -batch_size 128 -epsilon $EPS -num_attack_iterations 150\
                        -step_size 0.002 -attack_loss Shake
  wait
#  python main_attack.py -dataset cifar10 -robustness_level semi_robust -robust_model addeppalli2022_towards -num_adv_examples 10000\
#                        -cuda $CUDA -re_evaluation_mode True -batch_size 400 -epsilon $EPS
#  wait
#  python main_attack.py -dataset cifar10 -robustness_level semi_robust -robust_model engstrom2019 -num_adv_examples 10000\
#                        -cuda $CUDA -re_evaluation_mode True -batch_size 400 -epsilon $EPS
#  wait
done

