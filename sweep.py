import os
import argparse

parser = argparse.ArgumentParser(description="sweep parser")

parser.add_argument("--lr")
parser.add_argument("--batch-size")
parser.add_argument("--rehearsal-ratio")
parser.add_argument("--num-candidates")
parser.add_argument("--num-representatives")

args = parser.parse_args()

host = ""

command = (
    f"horovodrun -np 1 -H {host} python main.py "
    f"--backbone ptychonn "
    f"--model Er "
    f"--dataset ptycho "
    f"--dataset-dir /my-spack/datasets "
    f"--tasksets-config \"{{'scenario': 'reconstruction', 'num_tasks': 150}}\" "
    f"--buffer-config \"{{'implementation': 'standard', 'num_samples_per_representative': 3, 'num_candidates': {args.num_candidates}, 'num_representatives': {args.num_representatives}, 'rehearsal_ratio': {args.rehearsal_ratio}, 'provider': 'na+sm', 'discover_endpoints': True, 'cuda_rdma': False}}\" "
    f"--batch-size {args.batch_size} "
    f"--lr {args.lr} "
    f"--epochs 1 "
    f"--use-amp "
)

os.system(command)
