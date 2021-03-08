# GAN

A PyTorch implementation of [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) on MNIST.

## To run

`python3 train.py --epochs 50 --batch-size 128 --lr 2e-4 --beta-1 0.5 --num-workers 8 --save-model --model-path '../models/gan.pth'`

## To sample

`python3 sample.py --model-path '../models/gan.pth'`

## References

- [DCGAN Paper](https://arxiv.org/abs/1511.06434)
- [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial)
