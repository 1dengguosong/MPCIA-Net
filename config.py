
import os
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
import torch

def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epoch', type=int, default=15, help='batch_size')
    parser.add_argument('--train_path', type=str, default='data_dre/train.pkl')
    parser.add_argument('--dev_path', type=str, default='data_dre/dev.pkl')
    parser.add_argument('--test_path', type=str, default='data_dre/test.pkl')
    parser.add_argument('--model', type=str, default='./bert-base-uncased')
    parser.add_argument('--device', type=torch.device, default='cuda:0')

    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--ckpt_dir', type=Path, default='./models_r=0.99')
    parser.add_argument('--weight', type=float, default=1)
    parser.add_argument('--load_path', type=Path, default="./models_r=0.99/0.5376506024096386_epoch_6.pt")

    parser.add_argument('--num_attention_heads', type=int, default=6)
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.1)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument('--layer_norm_eps', type=float, default=1e-5)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--r', type=float, default=0.99)


    args = parser.parse_args()
    return args
.1
args = parse_arguments()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
