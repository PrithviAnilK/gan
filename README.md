# GAN

A Pytorch implementation of (Generative Adversarial Networks)[https://arxiv.org/abs/1406.2661] on MNIST dataset.

## To run

`python3 train.py --epochs 50 --batch-size 256 --lr 1e-4 --num-workers 4 --save-model --model-path '../models/gan.pth'`

## To sample

`python3 sample.py --model-path '../models/gan.pth'`
