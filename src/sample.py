import torch
from gan import Generator
from utils import get_config
from dataloader import rand_sampler
from torchvision.utils import save_image

def sample():
    args, device = get_config()
    generator = Generator()
    generator.to(device)
    generator.load_state_dict(torch.load(args.model_path))
    generator.eval()

    z = rand_sampler(1, device)
    g = generator(z)
    save_image(g[0], '../samples/sample.png')


if __name__ == "__main__":
    sample()