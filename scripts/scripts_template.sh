# The bash file should be named as task_dataset_model.sh
# Example: sl_cifar10_rn18.sh

# Benchmark: Your Benchmark
# Model: Your Model
# Method: Your Method
# Task: Your Task

python train.py \
    --cfg configs/pipeline/your_task/your_config.yaml \
    --opts device='cuda:0'