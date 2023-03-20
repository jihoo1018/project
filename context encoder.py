import tarfile
import zipfile

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import time


if torch.cuda.is_available():
  device = 'cuda:0'
else:
  device = 'cpu'

def filecheck():
  if not os.path.isfile("/content/voc_test_2007_tar"):
    tar = tarfile.open(r"C:\Users\AIA\project_final\content\Faster_RCNN\VOCtest_06-Nov-2007.tar")
    tar.extractall()
    tar.close()

  if not os.path.isfile("/content/city.zip"):
    city_zip = zipfile.ZipFile(r"C:\Users\AIA\project_final\content\city.zip")
    city_zip.extractall(r"C:\Users\AIA\project_final\images\city")
    city_zip.close()

  if not os.path.isfile("/content/nature.zip"):
    nature_zip = zipfile.ZipFile( r"C:\Users\AIA\project_final\content\nature.zip")
    nature_zip.extractall(r"C:\Users\AIA\project_final\images\nature")
    nature_zip.close()



def make_mask(size):
  label = 0
  while True:
    seg_list = os.listdir(r"C:\Users\AIA\project_final\content\VOCdevkit\VOC2007\SegmentationObject")
    seg_image = Image.open(r"C:\Users\AIA\project_final\content\VOCdevkit\VOC2007\SegmentationObject\\"+seg_list[np.random.randint(0,len(seg_list))])
    seg_image = seg_image.resize((size,size))
    np_seg = np.array(seg_image,dtype=np.uint8)
    labels = np.unique(np_seg)

    for lb in labels[1:-1]:
      if len(np.where(np_seg == lb)[0]) < (size**2)/4:
        label = lb
        break

    if label != 0:
      break

  np_seg = np.where(np_seg == label,1.0,0)
  np_seg = np.stack((np_seg,np_seg,np_seg),axis = 2)

  return np_seg


class Data(Dataset):
  def __init__(self, size=128):
    self.city_list = os.listdir(r"C:\Users\AIA\project_final\images\city")
    self.nature_list = os.listdir(r"C:\Users\AIA\project_final\images\nature")
    self.every_list = []
    self.to_tensor = transforms.ToTensor()
    self.size = size

    for x in self.city_list:
      self.every_list.append(r"C:\Users\AIA\project_final\images\city" + x)
    for x in self.nature_list:
      self.every_list.append(r"C:\Users\AIA\project_final\images\nature" + x)

    self.every_list.sort()

  def __len__(self):
    return len(self.every_list)

  def __getitem__(self, idx):
    mask = make_mask(self.size)
    image = Image.open(self.every_list[idx]).convert("RGB")

    image = image.crop((100, 200, image.size[0], image.size[1]))
    image = image.resize((self.size, self.size))
    image = np.array(image, dtype=np.uint8)
    image = image / 255

    # The missing region in the masked input image is filled with constant mean value.
    masked_image = (1 - mask) * image + mask * (np.zeros_like(image) + np.mean(image))

    return self.to_tensor(masked_image).type(torch.float32), self.to_tensor(mask).type(torch.float32), self.to_tensor(
      image).type(torch.float32)

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder,self).__init__()
    self.encoder = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=64,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv2d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(128),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(256),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv2d(in_channels=256,out_channels=512,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(512),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv2d(in_channels=512,out_channels=4000,kernel_size=4,stride=1,padding=0),
                                 nn.BatchNorm2d(4000),
                                 nn.LeakyReLU(0.2)
                                 )
  def forward(self, x):
    x = self.encoder(x)
    return x

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder,self).__init__()
    self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=4000,out_channels=512,kernel_size=4,stride=1,padding=0),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(),
                                 nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(256),
                                 nn.ReLU(),
                                 nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(),
                                 nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(),
                                 nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(),
                                 nn.ConvTranspose2d(in_channels=64,out_channels=3,kernel_size=4,stride=2,padding=1),
                                 nn.ReLU(),
                                 )
  def forward(self, x):
    x = self.decoder(x)
    return x


class ContextEncoder(nn.Module):
    def __init__(self):
        super(ContextEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        bottleneck = self.encoder(x)
        images = self.decoder(bottleneck)
        return images

class Discriminator(nn.Module):
    def __init__(self):
      super(Discriminator,self).__init__()
      self.encoder = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=64,kernel_size=4,stride=2,padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.LeakyReLU(0.2),
                                   nn.Conv2d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.LeakyReLU(0.2),
                                  nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.LeakyReLU(0.2),
                                  nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2,padding=1),
                                  nn.BatchNorm2d(256),
                                  nn.LeakyReLU(0.2),
                                  nn.Conv2d(in_channels=256,out_channels=512,kernel_size=4,stride=2,padding=1),
                                  nn.BatchNorm2d(512),
                                  nn.LeakyReLU(0.2),
                                  nn.Conv2d(in_channels=512,out_channels=1,kernel_size=4,stride=1,padding=0),
                                  nn.BatchNorm2d(1),
                                  nn.LeakyReLU(0.2),
                                  nn.Flatten(),
                                  nn.Sigmoid()
                                  )
    def forward(self, x):
      x = self.encoder(x)
      return x




if __name__ == '__main__':
    ContextEncoder().forward(nn)
