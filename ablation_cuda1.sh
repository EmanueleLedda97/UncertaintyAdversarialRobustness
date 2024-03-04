#!bin/bash


CUDA=1

for EPS in 0.0039 0.0078 0.012 0.0157 0.02 0.024 0.027 0.0313
do
#  # ----------------- STAB
#  python main_attack.py -dataset cifar10 -robustness_level semi_robust -robust_model sehwag2021 -num_adv_examples 5000\
#                        -cuda $CUDA -batch_size 128 -epsilon $EPS -num_attack_iterations 150\
#                        -step_size 0.002 -attack_loss Stab
#  wait
#  python main_attack.py -dataset cifar10 -robustness_level semi_robust -robust_model addepalli2022 -num_adv_examples 5000\
#                        -cuda $CUDA -batch_size 128 -epsilon $EPS -num_attack_iterations 150\
#                        -step_size 0.002 -attack_loss Stab
#  wait
#  python main_attack.py -dataset cifar10 -robustness_level semi_robust -robust_model addepalli2022_towards -num_adv_examples 5000\
#                        -cuda $CUDA -batch_size 128 -epsilon $EPS -num_attack_iterations 150\
#                        -step_size 0.002 -attack_loss Stab -re_evaluation_mode True
#  wait
#  python main_attack.py -dataset cifar10 -robustness_level semi_robust -robust_model engstrom2019 -num_adv_examples 5000\
#                        -cuda $CUDA -batch_size 128 -epsilon $EPS -num_attack_iterations 150\
#                        -step_size 0.002 -attack_loss Stab
#  wait

  # ----------------- SHAKE
#  python main_attack.py -dataset cifar10 -robustness_level semi_robust -robust_model sehwag2021 -num_adv_examples 5000\
#                        -cuda $CUDA -batch_size 128 -epsilon $EPS -num_attack_iterations 150\
#                        -step_size 0.002 -attack_loss Shake
#  wait
#  python main_attack.py -dataset cifar10 -robustness_level semi_robust -robust_model addepalli2022 -num_adv_examples 5000\
#                        -cuda $CUDA -batch_size 128 -epsilon $EPS -num_attack_iterations 150\
#                        -step_size 0.002 -attack_loss Shake
#  wait
#  python main_attack.py -dataset cifar10 -robustness_level semi_robust -robust_model addepalli2022_towards -num_adv_examples 5000\
#                        -cuda $CUDA -batch_size 128 -epsilon $EPS -num_attack_iterations 150\
#                        -step_size 0.002 -attack_loss Shake -re_evaluation_mode True
#  wait
#  python main_attack.py -dataset cifar10 -robustness_level semi_robust -robust_model engstrom2019 -num_adv_examples 5000\
#                        -cuda $CUDA -batch_size 128 -epsilon $EPS -num_attack_iterations 150\
#                        -step_size 0.002 -attack_loss Shake
#  wait

  # ----------------- AutoTarget
  python main_attack.py -dataset cifar10 -robustness_level semi_robust -robust_model sehwag2021 -num_adv_examples 8000\
                        -cuda $CUDA -batch_size 128 -epsilon $EPS -num_attack_iterations 150\
                        -step_size 0.002 -attack_loss AutoTarget
  wait
  python main_attack.py -dataset cifar10 -robustness_level semi_robust -robust_model addepalli2022 -num_adv_examples 8000\
                        -cuda $CUDA -batch_size 128 -epsilon $EPS -num_attack_iterations 150\
                        -step_size 0.002 -attack_loss AutoTarget
  wait
  python main_attack.py -dataset cifar10 -robustness_level semi_robust -robust_model addepalli2022_towards -num_adv_examples 8000\
                        -cuda $CUDA -batch_size 128 -epsilon $EPS -num_attack_iterations 150\
                        -step_size 0.002 -attack_loss AutoTarget
  wait
  python main_attack.py -dataset cifar10 -robustness_level semi_robust -robust_model engstrom2019 -num_adv_examples 8000\
                        -cuda $CUDA -batch_size 128 -epsilon $EPS -num_attack_iterations 150\
                        -step_size 0.002 -attack_loss AutoTarget
  wait
done

