import tarfile

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

def make_mask():
  # mask 만들기
  fig = plt.figure()

  item = 4
  original_list = os.listdir(r"C:\Users\AIA\project_final\content\VOCdevkit\VOC2007\JPEGImages")
  original_list.sort()
  original_image = Image.open(r"C:\Users\AIA\project_final\content\VOCdevkit\VOC2007\JPEGImages" + original_list[item])
  original_image = original_image.resize((256, 256))
  np_original = np.array(original_image, dtype=np.uint8) / 255
  fig.add_subplot(1, 2, 1)
  plt.imshow(np_original)
  plt.axis('off')

  seg_image = Image.open(r"C:\Users\AIA\project_final\content\VOCdevkit\VOC2007\SegmentationObject" + original_list[item][:-3] + "png")
  seg_image = seg_image.resize((256, 256))
  np_seg = np.array(seg_image, dtype=np.uint8)
  labels = np.unique(np_seg)
  print(labels)

  np_seg = np.where(np_seg == labels[1], 1.0, 0)
  np_seg = np.stack((np_seg, np_seg, np_seg), axis=2)
  print(np_seg.shape)

  masked_image = (1 - np_seg) * np_original + np_seg * (np.zeros_like(np_original) + np.mean(np_original))

  fig.add_subplot(1, 2, 2)
  plt.imshow(masked_image)
  plt.axis('off')

  plt.subplots_adjust(wspace=0.1, hspace=0)


def make_sample(size, batch_size):
  original_list = os.listdir(r"C:\Users\AIA\project_final\content\VOCdevkit\VOC2007\JPEGImages")

  while True:
    try:
      item = np.random.randint(0, len(original_list))
      original_image = Image.open(r"C:\Users\AIA\project_final\content\VOCdevkit\VOC2007\JPEGImages" + original_list[item]+'jpg')
      seg_image = Image.open(r"C:\Users\AIA\project_final\content\VOCdevkit\VOC2007\SegmentationObject" + original_list[item][:-3] + "png")
    except:
      continue

    original_image = original_image.resize((size, size))
    np_original = np.array(original_image, dtype=np.uint8) / 255

    seg_image = seg_image.resize((size, size))
    np_seg = np.array(seg_image, dtype=np.uint8)

    labels = np.unique(np_seg)
    label = 0
    for lb in labels[1:-1]:
      if len(np.where(np_seg == lb)[0]) < (size ** 2) / 4:
        label = lb
        break

    if label != 0:
      break

  np_seg = np.where(np_seg == labels[1], 1.0, 0)
  np_seg = np.stack((np_seg, np_seg, np_seg), axis=2)

  masked_image = (1 - np_seg) * np_original + np_seg * (np.zeros_like(np_original) + np.mean(np_original))

  return np_original, np_seg, masked_image

def sample():
  a, b, c = make_sample(128, 4)
  fig = plt.figure(figsize=(11, 11))
  fig.add_subplot(1, 3, 1)
  plt.imshow(a)
  fig.add_subplot(1, 3, 2)
  plt.imshow(b)
  fig.add_subplot(1, 3, 3)
  plt.imshow(c)

if __name__ == '__main__':
    make_mask()