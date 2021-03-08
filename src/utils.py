import argparse
import numpy as np
import random
import torch

def get_config():
    # Reference: https://github.com/pytorch/examples/blob/master/mnist/main.py
    parser = argparse.ArgumentParser(description='MNIST GAN')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N', help='input batch size for training (default: 256)')
    parser.add_argument('--num-workers', type=int, default=0, metavar='N', help='number of workers for training (default: 0)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 1e-4)')
    parser.add_argument('--beta-1', type=float, default=0.9, metavar='LR', help='beta_1 value in adam optim (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--save-model', action='store_true', default=False, help='for saving the current model')
    parser.add_argument('--model-path', type=str, default='../models/train_sample.pth', help='Path for saving the model')
    args = parser.parse_args()
    device = torch.device('cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    return args, device
