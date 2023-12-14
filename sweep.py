import os
import argparse

parser = argparse.ArgumentParser(description="sweep parser")

parser.add_argument("--lr", type=float, default=0.00088)
parser.add_argument("--batch-size", type=int, default=24)
parser.add_argument("--rehearsal-ratio", type=int, default=30)
parser.add_argument("--num-candidates", type=int, default=8)
parser.add_argument("--num-representatives", type=int, default=8)
parser.add_argument("--augmentations-offset", type=int, default=5)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--weight-decay", type=float, default=0.006)
parser.add_argument("--lr-schedule", type=str, default="exp_range_cyclic_lr")
parser.add_argument("--epoch-cycle-size", type=int, default=64)

args = parser.parse_args()

host = ""
sweep_conf = ""

baseline_scratch_ptycho_command = (
    f"horovodrun -np 1 -H {host} python main.py "
    f"--backbone ptychonn "
    f"--backbone-config \"{{'lr_schedule': '{args.lr_schedule}', 'epoch_cycle_size': {args.epoch_cycle_size}, 'weight_decay': {args.weight_decay}}}\" "
    f"--model Vanilla "
    f"--dataset ptycho "
    f"--dataset-dir /my-spack/datasets "
    f"--tasksets-config \"{{'scenario': 'reconstruction', 'num_tasks': 42, 'concatenate_tasksets': True}}\" "
    f"--batch-size {args.batch_size} "
    f"--lr {args.lr} "
    f"--epochs {args.epochs} "
    f"--use-amp "
    f"--load-checkpoint /root/distributed-continual-learning/checkpoint_initial_ptychonn_1.pth.tar "
)

ptycho_er_command = (
    f"horovodrun -np 1 -H {host} python main.py "
    f"--backbone ptychonn "
    f"--backbone-config \"{{'lr_schedule': '{args.lr_schedule}', 'epoch_cycle_size': {args.epoch_cycle_size}, 'weight_decay': {args.weight_decay}}}\" "
    f"--model Er "
    f"--dataset ptycho "
    f"--dataset-dir /my-spack/datasets "
    f"--tasksets-config \"{{'scenario': 'reconstruction', 'num_tasks': 42}}\" "
    f"--buffer-config \"{{'implementation': 'standard', 'num_samples_per_representative': 3, 'num_candidates': {args.num_candidates}, 'num_representatives': {args.num_representatives}, 'rehearsal_ratio': {args.rehearsal_ratio}, 'augmentations_offset': {args.augmentations_offset}, 'provider': 'na+sm', 'discover_endpoints': True, 'cuda_rdma': False}}\" "
    f"--batch-size {args.batch_size} "
    f"--lr {args.lr} "
    f"--epochs {args.epochs} "
    f"--use-amp "
    f"--load-checkpoint /root/distributed-continual-learning/checkpoint_initial_ptychonn_1.pth.tar "
)

command = {
    "baseline_scratch_ptycho": baseline_scratch_ptycho_command,
    "er_ptycho": ptycho_er_command,
}[sweep_conf]

os.system(command)
