import torch
from torch import nn
from torch import optim
from tqdm import tqdm 

from gan import Generator, Discriminator 
from dataloader import mnist_dataloader, rand_sampler
from torchvision.utils import save_image
from utils import get_config

def train():
    args, device = get_config()

    generator = Generator()
    discriminator = Discriminator()
    generator.to(device)
    discriminator.to(device)
    g_optim = optim.Adam(generator.parameters(), args.lr)
    d_optim = optim.Adam(discriminator.parameters(), args.lr)

    dataloader = mnist_dataloader(args)
    bce = nn.BCELoss()

    print('Training on device :', device.type)
    for e in range(args.epochs):
        running_d_loss = 0
        running_g_loss = 0
        count = 0
        runner = tqdm(enumerate(dataloader), total = len(dataloader), desc = 'Epoch: {}/{}'.format(e + 1, args.epochs))
        generator.train()
        
        for dex, batch in runner:
            x, _ = batch
            batch_size = x.size()[0]
            count += batch_size
            z = rand_sampler(batch_size, device)
            x = x.to(device)
            ones, zeros = torch.ones(batch_size, 1, device = device), torch.zeros(batch_size, 1, device = device)            
            
            d_optim.zero_grad(), g_optim.zero_grad()
            d_x = discriminator(x)
            g_z = generator(z)
            d_g = discriminator(g_z)
            d_x_loss = bce(d_x, ones) 
            d_g_loss = bce(d_g, zeros)
            d_loss = d_x_loss + d_g_loss
            d_loss.backward()
            d_optim.step()
            running_d_loss += d_loss.item()
            avg_d_loss = running_d_loss / count

            z = rand_sampler(batch_size, device)
            d_optim.zero_grad(), g_optim.zero_grad()
            g_z = generator(z)            
            d_g = discriminator(g_z)
            g_loss = bce(d_g, ones)
            g_loss.backward()
            g_optim.step()
            running_g_loss += g_loss.item()
            avg_g_loss = running_g_loss / count
            
            runner.set_postfix(d_loss = '{}'.format(avg_d_loss), g_loss = '{}'.format(avg_g_loss))
            
        
        z = rand_sampler(1, device)
        generator.eval()
        g = generator(z)
        save_image(g[0], '../samples/training/sample_e{}.png'.format(e + 1))

    if args.save_model :
        torch.save(generator.state_dict(), args.model_path)


if __name__ == "__main__":
    train()