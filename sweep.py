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
num_representatives = int(int(args.batch_size)*float(args.num_representatives))

host='chifflot-1.lille.grid5000.fr:2'

os.system("""mpiexec -np 2 -hosts """+host+""" python3 main.py --backbone resnet18 --model Derpp --dataset imagenet100 --tasksets-config "{'scenario': 'class', 'increment': 10, 'initial_increment': 10}" --buffer-config "{'num_representatives': """+str(num_representatives)+""", 'rehearsal_ratio': """+args.buffer_size+"""}" --epochs """+args.epochs+""" --dataset-dir /my-spack/datasets/ --batch-size """+args.batch_size+""" --alpha """+args.alpha+""" --beta """+args.beta+""" --lr """+args.lr)
