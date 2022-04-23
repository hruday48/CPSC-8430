

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import pickle
import PIL
from PIL import Image
import torchvision.utils as vutils










NOISE_DIM = 100
NUM_EPOCHS = 50
FEATURES_DISC = 64
FEATURES_GEN = 64
LEARNING_RATE = 0.0002
BATCH_SIZE = 100
IMAGE_SIZE = 64
CHANNELS_IMG = 3

gen_train=1
fakeSet=[]
realSet=[]


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = transforms.Compose(

        [
         transforms.Resize(IMAGE_SIZE),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transforms)


    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True, num_workers=2)


    gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5,.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5,.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
    writer_real = SummaryWriter(f"logs/real")
    writer_fake = SummaryWriter(f"logs/fake")
    step = 0

    gen.train()
    disc.train()

    for epoch in range(NUM_EPOCHS):

        for batch_idx, (real, Y_train) in enumerate(dataloader):
            if (Y_train.shape[0] < BATCH_SIZE):
                continue


            real = real.to(device)
            noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
            fake = gen(noise)


            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()


            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()


            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                      Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise)

                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:32], normalize=True
                    )



                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1

    for i in range(80):
        fake=gen(fixed_noise)
        fake=list(torch.split(fake,1,dim=0))
        real_batch=next(iter(dataloader))
        real=real_batch[0].to(device)[:64]
        real=list(torch.split(real,1,dim=0))
        for f in fake:
            fakeSet.append(f)
        for r in real:
            realSet.append(r)


    for index,i in enumerate(fakeSet):
        vutils.save_image(i,"fake_images/fake"+str(index)+".jpeg")
    for index,i in enumerate(realSet):
        vutils.save_image(i,"real_images/real"+str(index)+".jpeg")
