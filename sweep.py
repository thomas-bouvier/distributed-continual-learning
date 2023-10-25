import os
import argparse

parser = argparse.ArgumentParser(description="sweep parser")

parser.add_argument("--epochs")
parser.add_argument("--lr")
parser.add_argument("--alpha")
parser.add_argument("--beta")
parser.add_argument("--batch-size")
parser.add_argument("--buffer-size")
parser.add_argument("--num-representatives")

args = parser.parse_args()
num_representatives = int(int(args.batch_size) * float(args.num_representatives))

host = ""

os.system(
    f"horovodrun -np 2 -H "
    ""
    + host
    + """ python main.py --backbone resnet18 --model Derpp --model-config "{'alpha': """
    + args.alpha
    + """, 'beta': """
    + args.beta
    + """}" --dataset imagenet100 --tasksets-config "{'scenario': 'class', 'increment': 10, 'initial_increment': 10}" --buffer-config "{'num_representatives': """
    + str(num_representatives)
    + """, 'rehearsal_ratio': """
    + args.buffer_size
    + """}" --epochs """
    + args.epochs
    + """ --dataset-dir /my-spack/datasets/ --batch-size """
    + args.batch_size
    + """ --lr """
    + args.lr
)
