import torch.nn as nn
from torch.distributions.normal import Normal
import functools
import operator
import torch.nn.functional as F
import torch
import random
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import cv2
import os, random
from PIL import Image
import numpy as np

import argparse

def add_noise(images, mean=0, std=0.1):
    normal_dst = Normal(mean, std)
    noise = normal_dst.sample(images.shape)
    noisy_image = noise + images
    return noisy_image

class Auto_Encoder_Model(nn.Module):
    def __init__(self, im_c=3, activation=nn.ReLU()):
        super(Auto_Encoder_Model, self).__init__()
        self.act = activation 
        # Encoder
        self.conv1 = nn.Conv2d(im_c, 128, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)

        # Decoder
        self.tran_conv1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2)
        self.tran_conv2 = nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, output_padding=[1, 1])
        self.tran_conv3 = nn.ConvTranspose2d(128, im_c, kernel_size=8, stride=4)
        

    def forward_pass(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x1))
        x3 = self.act(self.conv3(x2))
        return x3

    def reconstruct_pass(self, x):
        output = self.act(self.tran_conv1(x))
        output = self.act(self.tran_conv2(output))
        output = self.act(self.tran_conv3(output))
        return output

    def forward(self, x):
        output = self.forward_pass(x)
        output = self.reconstruct_pass(output)
        return output

def train(args):
    #train AE
    device = 'cuda:0'
    im_channel = args.im_channels
    im_w = args.sizes
    model = Auto_Encoder_Model(im_c=im_channel).to(device)
    
    # model.load_state_dict(torch.load('./ae/ae_full_prelu.pth'))
    ae_criterion = torch.nn.MSELoss().to(device)
    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))

    #load data
    # ToTensor already divided the image by 255
    batch_size = 16
    dataset = datasets.ImageFolder(args.data,\
         transform=transforms.Compose([transforms.Resize((224, 224)), transforms.Grayscale(), transforms.ToTensor()]))
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    #train
    epochs = 10
    for epoch in range(0, epochs):
        current_loss = 0
        for i,data in enumerate(data_loader,0):
            input, _ = data
            input = input.to(device)
            
            optim.zero_grad()
            encoder = model.forward_pass(input)
            decoder = model.reconstruct_pass(encoder)
            loss = ae_criterion(decoder, input)

            loss.backward()
            optim.step()

            loss = loss.item()
            current_loss += loss
        print('{} loss : {}'.format(epoch+1, batch_size*current_loss/len(data_loader)))

    torch.save(model.state_dict(),'simple_ae_{}_depth_{}_size.pth'.format(im_channel, im_w))

def test(args):
    device = 'cuda:0'
    model = Auto_Encoder_Model(im_c=args.im_channels).to(device)
    weight_path = args.weight_path
    print('load initial weights from: %s'%(weight_path))
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    folder = args.data + '/images/'
    files = os.listdir(folder)
    f = random.choice(files)

    image_path = folder + f
    if args.im_channels == 1:
        image = cv2.imread(image_path, 0)
        image = cv2.resize(image, (224, 224))
    else:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # normalize
    image_pil = Image.fromarray(image)
    image = image/255
    # print(image)
    
    
    # to_tensor = transforms.ToTensor()
    image = torch.Tensor(image)
    if args.im_channels == 1:
        image = image.unsqueeze(-1)
    image = image.permute(-1,0,1)
    image = image.unsqueeze(0)
    
    t = time.time()
    
    image = image.to(device)
    encoder = model.forward_pass(image)
    decoder = model.reconstruct_pass(encoder)
    decoder = decoder.squeeze(0)
    decoder = decoder.permute(1, 2, 0)
    
    f = plt.figure()
    f.add_subplot(1,1, 1) #1,2,1
    plt.imshow(decoder.cpu().detach().numpy())
    cv2.imshow("a", decoder.cpu().detach().numpy())
    cv2.waitKey(0)
    print(time.time()-t)
    plt.show()

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1, help="1: Train, 0: Evaluate")
    parser.add_argument("--im_channels", type=int, default=1, help="3: RGB, 1: Depth")
    parser.add_argument("--sizes", type=int, default=224, help="Image width (height)")
    parser.add_argument("--data", type=str, default="/home/tinvn/TIN/Agileflight_pictures/Depth/", help="data folder")
    parser.add_argument("--weight_path", type=str, default="simple_ae_1_depth_224_size.pth", help="weight path")
    parser.add_argument("--render", type=int, default=0, help="1: visualize")
    return parser

if __name__ == "__main__":
    args = parser().parse_args()
    if args.train:
        train(args)
    else:
        test(args)