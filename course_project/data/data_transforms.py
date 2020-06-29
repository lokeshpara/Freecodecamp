from albumentations import (
	Compose,
    HorizontalFlip,
    Normalize,
    RandomCrop,
    PadIfNeeded,
    RGBShift,
    Rotate
)
from albumentations.pytorch import ToTensor
import numpy as np
import torchvision.transforms as transforms

def albumentations_transforms(p=1.0, is_train=False):
	# Mean and standard deviation of train dataset
	mean = np.array([0.4914, 0.4822, 0.4465])
	std = np.array([0.2023, 0.1994, 0.2010])
	transforms_list = []
	# Use data aug only for train data
	if is_train:
		transforms_list.extend([
			PadIfNeeded(min_height=72, min_width=72, p=1.0),
			RandomCrop(height=64, width=64, p=1.0),
			HorizontalFlip(p=0.25),
			Rotate(limit=15, p=0.25),
			RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.25),
			#CoarseDropout(max_holes=1, max_height=32, max_width=32, min_height=8,
						#min_width=8, fill_value=mean*255.0, p=0.5),
		])
	transforms_list.extend([
		Normalize(
			mean=mean,
			std=std,
			max_pixel_value=255.0,
			p=1.0
		),
		ToTensor()
	])
	data_transforms = Compose(transforms_list, p=p)
	return lambda img: data_transforms(image=np.array(img))["image"]

def torch_transforms(is_train=False):
	# Mean and standard deviation of train dataset
	mean = (0.4914, 0.4822, 0.4465)
	std = (0.2023, 0.1994, 0.2010)
	transforms_list = []
	# Use data aug only for train data
	if is_train:
		transforms_list.extend([
			transforms.RandomCrop(64, padding=4),
			transforms.RandomHorizontalFlip(),
		])
	transforms_list.extend([
		transforms.ToTensor(),
		transforms.Normalize(mean, std),
	])
	if is_train:
		transforms_list.extend([
			transforms.RandomErasing(0.25)
		])
	return transforms.Compose(transforms_list)
