#!bin/bash




for EPS in 0.0156 0.0118 0.0078 0.0039
do
    python main_attack.py -dataset imagenet -robustness_level semi_robust -robust_model Liu2023swinL -num_adv_examples 10000 -cuda 1 -re_evaluation_mode False -batch_size 100 -epsilon $EPS
    wait

    python main_attack.py -dataset imagenet -robustness_level semi_robust -robust_model Liu2023convNextB -num_adv_examples 10000 -cuda 1 -re_evaluation_mode False -batch_size 100 -epsilon $EPS
    wait

    python main_attack.py -dataset imagenet -robustness_level semi_robust -robust_model Liu2023convNextL -num_adv_examples 10000 -cuda 1 -re_evaluation_mode False -batch_size 100 -epsilon $EPS
    wait
done

for EPS in 0.0156 0.0118 0.0078 0.0039
do
    python main_attack.py -dataset imagenet -robustness_level semi_robust -robust_model Liu2023swinB -num_adv_examples 10000 -cuda 1 -re_evaluation_mode False -batch_size 100 -epsilon $EPS
    wait
done


# for EPS in 0.0314 0.0274 0.0235 0.0196 0.0156 0.0118 0.0078 0.0039
# do
#     python main_attack.py -dataset cifar10 -backbone resnet18 -robustness_level naive_robust -uq_technique None -num_adv_examples 5000 -cuda 1 -re_evaluation_mode True -batch_size 1000 -epsilon $EPS
#     wait

#     python main_attack.py -dataset cifar10 -backbone resnet34 -robustness_level naive_robust -uq_technique None -num_adv_examples 5000 -cuda 1 -re_evaluation_mode True -batch_size 1000 -epsilon $EPS
#     wait

#     python main_attack.py -dataset cifar10 -backbone resnet50 -robustness_level naive_robust -uq_technique None -num_adv_examples 5000 -cuda 1 -re_evaluation_mode True -batch_size 1000 -epsilon $EPS
#     wait
# done