"""
This is a slight modification of Shubham Chandel's implementation of a
variational autoencoder in PyTorch.

Source: https://github.com/sksq96/pytorch-vae
"""
import cv2
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

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
        bce = F.binary_cross_entropy(recon, actual, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
        return bce + kld * kld_weight


if __name__ == "__main__":
    # with 0~255
    dataset = datasets.ImageFolder('/home/tinvn/TIN/Agileflight_pictures/RGB/', \
        transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
    # data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    bsz = 32
    lr = 1e-4

    n = len(dataset)
    indices = np.random.permutation(n)
    thres = int(n * 0.9)
    train_indices, valid_indices = indices[:thres], indices[thres:]
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=bsz, 
                                            sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=bsz,
                                                    sampler=valid_sampler)


    
    vae = VAE(im_c=3, im_h=224, im_w=224, z_dim=64).to(device)
    optim = torch.optim.Adam(vae.parameters(), lr=lr)
    num_epochs = 1000
    best_loss = 1e10
    for epoch in range(num_epochs):
        
        train_loss = []
        # Train
        vae.train()
        # Train:   
        for batch_index, data in enumerate(train_loader):
            input, _ = data
            input = input.to(device)
            
            loss = vae.loss(input, *vae(input))
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)
        test_loss = []
        # Eval
        vae.eval()
        for batch_index, data in enumerate(validation_loader):
            input, _ = data
            # input = input.view(input.size(0), input.size(2), input.size(3), -1)
            input = input.to(device)

            loss = vae.loss(input, *vae(input), kld_weight=0.0)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)
        print(f"#{epoch + 1} train_loss: {train_loss:.6f}, test_loss: {test_loss:.6f}")
        if test_loss < best_loss and epoch > num_epochs / 10:
            best_loss = test_loss
            print(f"save model at epoch #{epoch + 1}")
            torch.save(vae.state_dict(), "vae.pth")
        
