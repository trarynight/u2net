# data loader
from __future__ import print_function, division
import glob
import torch
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import cv2
from albumentations import (
    Compose,
	RandomCrop,
	Resize,
	SmallestMaxSize,
    RandomResizedCrop,
)

def generate_transforms():
    train_transform = Compose(
        [	
			SmallestMaxSize(max_size=320),
			RandomCrop(height=288, width=288)
        ]
    )
    return train_transform

class SalObjDataset(Dataset):
	def __init__(self, img_name_list, lbl_name_list, transform=None):
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self, idx):
		image = cv2.cvtColor(cv2.imread(self.image_name_list[idx]), cv2.COLOR_BGR2RGB)
		imname = self.image_name_list[idx]
		imidx = np.array([idx])

		if(0==len(self.label_name_list)):
			label = np.zeros(image.shape[:2])
			label = np.expand_dims(label, -1)
		else:
			label = cv2.imread(self.label_name_list[idx], 0)
			label = np.expand_dims(label, -1)


		sample = {'imidx':imidx, 'image':image, 'label':label}

		if self.transform is not None:
			seed = np.random.randint(2147483647)
			random.seed(seed)
			image = self.transform(image=image)["image"]
			image = transforms.ToTensor()(image)
			image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
			random.seed(seed)
			label = self.transform(image=label)["image"]
			label = transforms.ToTensor()(label)
			sample = {'imidx':imidx, 'image':image, 'label':label}

		return sample


class SalObjDatasetT(Dataset):
	def __init__(self, img_name_list, lbl_name_list, transform=None):
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self, idx):
		image = cv2.cvtColor(cv2.imread(self.image_name_list[idx]), cv2.COLOR_BGR2RGB)
		h, w = image.shape[:2]
		image = Image.fromarray(image)
		imname = self.image_name_list[idx]
		imidx = np.array([idx])

		if(0==len(self.label_name_list)):
			label = np.zeros(image.shape[:2])
			label = np.expand_dims(label, -1)
			label = Image.fromarray(np.uint8(label))
		else:
			label = np.zeros([h, w, 1])
			label_img = cv2.imread(self.label_name_list[idx])
			label = label_img[:, :, 0]
			label = Image.fromarray(np.uint8(label))


		sample = {'imidx':imidx, 'image':image, 'label':label}

		if self.transform is not None:
			seed = np.random.randint(2147483647)
			random.seed(seed)
			image = self.transform(image)
			# image = transforms.ToTensor()(image)
			image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
			random.seed(seed)
			label = self.transform(label)
			# label = transforms.ToTensor()(label)
			sample = {'imidx':imidx, 'image':image, 'label':label}

		return sample