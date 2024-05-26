import os
import argparse

parser = argparse.ArgumentParser(description="sweep parser")

parser.add_argument("--lr", type=float, default=0.00088)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--rehearsal-ratio", type=int, default=30)
parser.add_argument("--num-candidates", type=int, default=8)
parser.add_argument("--num-representatives", type=int, default=24)
parser.add_argument("--augmentations-offset", type=int, default=16)
parser.add_argument("--soft-augmentations-offset", type=int, default=25)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--weight-decay", type=float, default=0)
parser.add_argument("--lr-schedule", type=str, default="exp_range_cyclic_lr")
parser.add_argument("--step-cycle-size", type=int, default=2944)
parser.add_argument("--alpha", type=float, default=0.8)
parser.add_argument("--beta", type=float, default=1.0)

args = parser.parse_args()

host = ""
sweep_conf = ""

er_command = (
    f"horovodrun -np 8 -H {host} python main.py "
    f"--backbone resnet50 "
    f"--model Er "
    f"--dataset imagenet_blurred "
    f"--dataset-dir /my-spack/datasets "
    f"--tasksets-config \"{{'scenario': 'class', 'initial_increment': 250, 'increment': 250}}\" "
    f"--buffer-config \"{{'implementation': 'standard', 'num_samples_per_representative': 1, 'num_candidates': {args.num_candidates}, 'num_representatives': {args.num_representatives}, 'rehearsal_ratio': 30, 'augmentations_offset': {args.augmentations_offset}, 'provider': 'na+sm', 'discover_endpoints': True, 'cuda_rdma': False}}\" "
    f"--batch-size 56 "
    f"--lr 0.0125 "
    f"--epochs 30 "
    f"--warmup-epochs 5 "
    f"--use-amp "
    f"--use-dali "
    f"--load-checkpoint /root/distributed-continual-learning/checkpoint_initial_resnet50_imagenet_1.pth.tar "
)

der_command = (
    f"horovodrun -np 8 -H {host} python main.py "
    f"--backbone resnet50 "
    f"--model Der "
    f"--model-config \"{{'alpha': {args.alpha}}}\" "
    f"--dataset imagenet_blurred "
    f"--dataset-dir /my-spack/datasets "
    f"--tasksets-config \"{{'scenario': 'class', 'initial_increment': 250, 'increment': 250}}\" "
    f"--buffer-config \"{{'implementation': 'standard', 'num_samples_per_representative': 1, 'num_samples_per_activation': 1, 'activation_shape': [1000], 'num_candidates': {args.num_candidates}, 'num_representatives': {args.num_representatives}, 'rehearsal_ratio': 30, 'augmentations_offset': {args.augmentations_offset}, 'provider': 'na+sm', 'discover_endpoints': True, 'cuda_rdma': False}}\" "
    f"--batch-size 56 "
    f"--lr 0.0125 "
    f"--epochs 30 "
    f"--warmup-epochs 5 "
    f"--use-amp "
    f"--use-dali "
    f"--load-checkpoint /root/distributed-continual-learning/checkpoint_initial_resnet50_imagenet_1.pth.tar "
)

baseline_scratch_ptycho_command = (
    f"horovodrun -np 8 -H {host} python main.py "
    f"--backbone ptychonn "
    f"--backbone-config \"{{'lr_schedule': '{args.lr_schedule}', 'step_cycle_size': {args.step_cycle_size}, 'weight_decay': {args.weight_decay}}}\" "
    f"--model Vanilla "
    f"--dataset ptycho "
    f"--dataset-dir /my-spack/datasets "
    f"--tasksets-config \"{{'scenario': 'reconstruction', 'num_tasks': 156, 'concatenate_tasksets': True}}\" "
    f"--batch-size {args.batch_size} "
    f"--eval-batch-size 5 "
    f"--lr {args.lr} "
    f"--epochs {args.epochs} "
    f"--use-amp "
    f"--use-dali "
    f"--load-checkpoint /root/distributed-continual-learning/checkpoint_initial_ptychonn_1.pth.tar "
)

er_ptycho_command = (
    f"horovodrun -np 8 -H {host} python main.py "
    f"--backbone ptychonn "
    f"--backbone-config \"{{'lr_schedule': '{args.lr_schedule}', 'step_cycle_size': {args.step_cycle_size}, 'weight_decay': {args.weight_decay}}}\" "
    f"--model Er "
    f"--dataset ptycho "
    f"--dataset-dir /my-spack/datasets "
    f"--tasksets-config \"{{'scenario': 'reconstruction', 'num_tasks': 156}}\" "
    f"--buffer-config \"{{'implementation': 'standard', 'num_samples_per_representative': 3, 'num_candidates': {args.num_candidates}, 'num_representatives': {args.num_representatives}, 'rehearsal_ratio': {args.rehearsal_ratio}, 'augmentations_offset': {args.augmentations_offset}, 'provider': 'na+sm', 'discover_endpoints': True, 'cuda_rdma': False}}\" "
    f"--batch-size {args.batch_size} "
    f"--eval-batch-size 5 "
    f"--lr {args.lr} "
    f"--epochs {args.epochs} "
    f"--use-amp "
    f"--use-dali "
    f"--load-checkpoint /root/distributed-continual-learning/checkpoint_initial_ptychonn_1.pth.tar "
)

derpp_ptycho_command = (
    f"horovodrun -np 8 -H {host} python main.py "
    f"--backbone ptychonn "
    f"--backbone-config \"{{'lr_schedule': '{args.lr_schedule}', 'step_cycle_size': {args.step_cycle_size}, 'weight_decay': {args.weight_decay}}}\" "
    f"--model Derpp "
    f"--model-config \"{{'alpha': {args.alpha}, 'beta': {args.beta}}}\" "
    f"--dataset ptycho "
    f"--dataset-dir /my-spack/datasets "
    f"--tasksets-config \"{{'scenario': 'reconstruction', 'num_tasks': 156}}\" "
    f"--buffer-config \"{{'implementation': 'standard', 'num_samples_per_representative': 3, 'num_candidates': {args.num_candidates}, 'num_representatives': {args.num_representatives}, 'rehearsal_ratio': {args.rehearsal_ratio}, 'augmentations_offset': {args.augmentations_offset}, 'provider': 'na+sm', 'discover_endpoints': True, 'cuda_rdma': False}}\" "
    f"--batch-size {args.batch_size} "
    f"--eval-batch-size 5 "
    f"--lr {args.lr} "
    f"--epochs {args.epochs} "
    f"--use-amp "
    f"--use-dali "
    f"--load-checkpoint /root/distributed-continual-learning/checkpoint_initial_ptychonn_1.pth.tar "
)

beta_derpp_ptycho_command = (
    f"horovodrun -np 8 -H {host} python main.py "
    f"--backbone ptychonn "
    f"--backbone-config \"{{'lr_schedule': '{args.lr_schedule}', 'step_cycle_size': {args.step_cycle_size}, 'weight_decay': {args.weight_decay}}}\" "
    f"--model Derpp "
    f"--model-config \"{{'alpha': {args.alpha}, 'beta': {args.beta}}}\" "
    f"--dataset ptycho "
    f"--dataset-dir /my-spack/datasets "
    f"--tasksets-config \"{{'scenario': 'reconstruction', 'num_tasks': 156}}\" "
    f"--buffer-config \"{{'implementation': 'standard', 'num_samples_per_representative': 3, 'num_candidates': {args.num_candidates}, 'num_representatives': {args.num_representatives}, 'rehearsal_ratio': {args.rehearsal_ratio}, 'augmentations_offset': {args.augmentations_offset}, 'soft_augmentations_offset': {args.soft_augmentations_offset}, 'provider': 'na+sm', 'discover_endpoints': True, 'cuda_rdma': False}}\" "
    f"--batch-size {args.batch_size} "
    f"--eval-batch-size 5 "
    f"--lr {args.lr} "
    f"--epochs 1 "
    f"--use-amp "
    f"--use-dali "
    f"--load-checkpoint /root/distributed-continual-learning/checkpoint_initial_ptychonn_1.pth.tar "
)

command = {
    "er": er_command,
    "der": der_command,
    "baseline_scratch_ptycho": baseline_scratch_ptycho_command,
    "er_ptycho": er_ptycho_command,
    "derpp_ptycho": derpp_ptycho_command,
    "beta_derpp_ptycho": beta_derpp_ptycho_command,
}[sweep_conf]

os.system(command)
