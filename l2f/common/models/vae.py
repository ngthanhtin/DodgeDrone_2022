"""
This is a slight modification of Shubham Chandel's implementation of a
variational autoencoder in PyTorch.

Source: https://github.com/sksq96/pytorch-vae
"""
import os
import cv2
import random
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import argparse

class VAE(nn.Module):
    """Input should be (bsz, C, H, W) where C=3, H=42, W=144"""

    def __init__(self, im_c=3, im_h=144, im_w=144, z_dim=32):
        super().__init__()

        self.im_c = im_c
        self.im_h = im_h
        self.im_w = im_w

        encoder_list = [
            nn.Conv2d(im_c, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        ]

        # encoder_list = [
        #     nn.Conv2d(im_c, 128, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # ]

        self.encoder = nn.Sequential(*encoder_list)
        sample_img = torch.zeros([1, im_c, im_h, im_w])
        em_shape = nn.Sequential(*encoder_list[:-1])(sample_img).shape[1:]
        h_dim = np.prod(em_shape)
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, em_shape),
            nn.ConvTranspose2d(
                em_shape[0],
                128,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=(0, 0),
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1, output_padding=(0, 0)
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(32, im_c, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        # self.decoder = nn.Sequential(
        #     nn.Unflatten(1, em_shape),
        #     nn.ConvTranspose2d(
        #         em_shape[0],
        #         64,
        #         kernel_size=4,
        #         stride=2,
        #     ),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, output_padding=[1, 1]),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(
        #         128, im_c, kernel_size=8, stride=4
        #     ),
        #     nn.Sigmoid(),
        # )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size(), device=mu.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def encode_raw(self, x: np.ndarray, device) -> np.ndarray:
        # assume x is RGB image with shape (bsz, H, W, 3) or depth (bsz, H, W, 1)
        p = np.zeros([x.shape[0], 224, 224, self.im_c], np.float)

        for i in range(x.shape[0]):
            if self.im_c == 3: #rgb
                p[i] = cv2.resize(x[i], (224, 224), interpolation=cv2.INTER_AREA)
            if self.im_c == 1: # depth
                p[i] = np.expand_dims(cv2.resize(x[i], (224, 224)), -1)
        x = p.transpose(0, 3, 1, 2)
        x = torch.as_tensor(x, device=device, dtype=torch.float)
        v = self.representation(x)
        
        return v, v.detach().cpu().numpy()

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        return self.decoder(z)

    def forward(self, x):
        # expects (N, C, H, W)
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

    def loss(self, actual, recon, mu, logvar, kld_weight=1.0):
        bce = F.binary_cross_entropy(recon, actual, reduction="mean")        
        kld = -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
        
        return bce + kld * kld_weight

def test(args):
    device = 'cuda:2'
    model = VAE(im_c=args.im_channels, im_h=args.sizes, im_w=args.sizes, z_dim=args.z_dim).to(device)
    weight_path = args.load_weight_path
    print('load initial weights from: %s'%(weight_path))
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    folder = args.data + "/images/"
    files = os.listdir(folder)
    f = random.choice(files)

    image_path = folder + f
    image = cv2.imread(image_path)
    image = cv2.resize(image, (args.sizes, args.sizes))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_pil = Image.fromarray(image)
    image = image/255
    # print(image)
    
    
    # to_tensor = transforms.ToTensor()
    image = torch.Tensor(image)
    image = image.permute(-1,0,1)
    image = image.unsqueeze(0)
    
    image = image.to(device)
    z, mu, logvar = model.forward(image)
    decoder = z.squeeze()
    decoder = decoder.permute(1, 2, 0)
    
    f = plt.figure()
    f.add_subplot(1,1, 1) #1,2,1
    plt.imshow(decoder.cpu().detach().numpy())

    plt.show()

def train(args):
    #train AE
    device = 'cuda:2'
    img_w, img_h = args.sizes, args.sizes
    z_dim = args.z_dim
    im_channel = args.im_channels
    model = VAE(im_c=im_channel, im_h=img_h, im_w=img_w, z_dim=z_dim).to(device)
    
    if args.continue_training:
        model.load_state_dict(torch.load(args.load_weight_path))
    #hyper params
    lr = 1e-3
    batch_size = 32
    epochs = 1000
    
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    #load data
    # ToTensor already divided the image by 255
    dataset = datasets.ImageFolder(args.data, \
         transform=transforms.Compose([transforms.Resize((img_w, img_h)), transforms.ToTensor()]))
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    #
    for epoch in range(0, epochs):
        current_loss = 0
        for i,data in enumerate(data_loader,0):
            input, _ = data
            input = input.to(device)
            
            optim.zero_grad()
            decode, mu, logvar = model.forward(input)
            
            loss = model.loss(input, decode, mu, logvar, kld_weight=args.kld_weight) 
            loss.backward()
            optim.step()
            # scheduler.step()

            loss = loss.item()
            current_loss += loss
        print('{} loss : {}'.format(epoch+1, batch_size*current_loss/len(data_loader)))

    torch.save(model.state_dict(), "{}".format(args.save_weight_path))

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1, help="1: Train, 0: Evaluate")
    parser.add_argument("--im_channels", type=int, default=3, help="3: RGB, 1: Depth")
    parser.add_argument("--sizes", type=int, default=144, help="Image width (height)")
    parser.add_argument("--z-dim", type=int, default=32, help="Latent dims")
    parser.add_argument("--kld-weight", type=float, default=0.0, help="KL Divergence weight")
    parser.add_argument("--data", type=str, default="/home/tinvn/TIN/Agileflight_pictures/RGB/", help="data folder")
    parser.add_argument("--render", type=int, default=0, help="1: visualize")
    parser.add_argument("--load_weight_path", type=str, default="vae224.pth", help="weight path")
    parser.add_argument("--save_weight_path", type=str, default="vae224.pth", help="weight path")
    parser.add_argument("--continue-training", type=int, default="0", help="continue training with a pretrained weight")
    return parser

if __name__ == "__main__":
    args = parser().parse_args()
    if args.train:
        train(args)
    else:
        test(args)
