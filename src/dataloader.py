import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def mnist_dataloader(args):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5), (0.5)),
    ])

    dataset = datasets.MNIST('../dataset/MNIST', train = True, download = True, transform =  transform)
    dataloader = DataLoader(dataset, shuffle = True, batch_size = args.batch_size, num_workers = args.num_workers, pin_memory = True)
    return dataloader


def rand_sampler(batch_size, device):
    z = torch.randn(batch_size, 100, device = device)
    return z