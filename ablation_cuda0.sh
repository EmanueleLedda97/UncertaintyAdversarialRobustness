#!bin/bash

# # ---------- Transformers ----------
# python main_attack.py -dataset imagenet -backbone ConvNeXt-L -robustness_level naive_robust -uq_technique None -num_adv_examples 10000 -cuda 0 -re_evaluation_mode True -batch_size 20 -epsilon 0.0156
# wait
# python main_attack.py -dataset imagenet -backbone ConvNeXt-B -robustness_level naive_robust -uq_technique None -num_adv_examples 10000 -cuda 0 -re_evaluation_mode True -batch_size 20 -epsilon 0.0156
# wait
# python main_attack.py -dataset imagenet -backbone Swin-B -robustness_level naive_robust -uq_technique None -num_adv_examples 10000 -cuda 0 -re_evaluation_mode True -batch_size 20 -epsilon 0.0156
# wait


# # 3/255 -> 1/255
# for EPS in 0.0118 0.0078 0.0039
# do
#     # ---------- ResNets ----------
#     # # NAIVE MODELS
#     # python main_attack.py -dataset imagenet -backbone resnet18 -robustness_level naive_robust -uq_technique None -num_adv_examples 10000 -cuda 0 -re_evaluation_mode True -batch_size 400 -epsilon $EPS
#     # wait
#     # python main_attack.py -dataset imagenet -backbone resnet50 -robustness_level naive_robust -uq_technique None -num_adv_examples 10000 -cuda 0 -re_evaluation_mode True -batch_size 400 -epsilon $EPS
#     # wait

#     # # ROBUST MODELS
#     # python main_attack.py -dataset imagenet -robustness_level semi_robust -robust_model salman2020R18 -num_adv_examples 10000 -cuda 0 -re_evaluation_mode True -batch_size 400 -epsilon $EPS
#     # wait
#     # python main_attack.py -dataset imagenet -robustness_level semi_robust -robust_model wong2020 -num_adv_examples 10000 -cuda 0 -re_evaluation_mode True -batch_size 400 -epsilon $EPS
#     # wait
#     # python main_attack.py -dataset imagenet -robustness_level semi_robust -robust_model engstrom2019imgnet -num_adv_examples 10000 -cuda 0 -re_evaluation_mode True -batch_size 400 -epsilon $EPS
#     # wait
#     # python main_attack.py -dataset imagenet -robustness_level semi_robust -robust_model salman2020R50 -num_adv_examples 10000 -cuda 0 -re_evaluation_mode True -batch_size 400 -epsilon $EPS
#     # wait

#     # ---------- Transformers ----------
#     python main_attack.py -dataset imagenet -backbone ConvNeXt-L -robustness_level naive_robust -uq_technique None -num_adv_examples 10000 -cuda 0 -re_evaluation_mode True -batch_size 100 -epsilon $EPS
#     wait
#     python main_attack.py -dataset imagenet -backbone ConvNeXt-B -robustness_level naive_robust -uq_technique None -num_adv_examples 10000 -cuda 0 -re_evaluation_mode True -batch_size 100 -epsilon $EPS
#     wait
#     python main_attack.py -dataset imagenet -backbone Swin-B -robustness_level naive_robust -uq_technique None -num_adv_examples 10000 -cuda 0 -re_evaluation_mode True -batch_size 100 -epsilon $EPS
#     wait

# done



for EPS in 0.0156 0.0118 0.0078 0.0039
do
    python main_attack.py -dataset imagenet -backbone ConvNeXt-L -robustness_level naive_robust -uq_technique None -num_adv_examples 10000 -cuda 0 -re_evaluation_mode True -batch_size 50 -epsilon $EPS -attack_loss Shake
    wait

    python main_attack.py -dataset imagenet -backbone ConvNeXt-B -robustness_level naive_robust -uq_technique None -num_adv_examples 10000 -cuda 0 -re_evaluation_mode True -batch_size 50 -epsilon $EPS -attack_loss Shake
    wait

    python main_attack.py -dataset imagenet -backbone Swin-B -robustness_level naive_robust -uq_technique None -num_adv_examples 10000 -cuda 0 -re_evaluation_mode True -batch_size 50 -epsilon $EPS -attack_loss Shake
    wait
done