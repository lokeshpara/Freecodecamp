import torch
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import cv2

def has_cuda():
	return torch.cuda.is_available()

def which_device():
	return torch.device("cuda" if has_cuda() else "cpu")

def init_seed(args):
	torch.manual_seed(args.seed)

	if has_cuda():
		print("CUDA Available")
		torch.cuda.manual_seed(args.seed)

def show_model_summary(model, input_size):
	print(summary(model, input_size=input_size))

def imshow(img, title):
	img = denormalize(img)
	npimg = img.numpy()
	fig = plt.figure(figsize=(15,7))
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.title(title)


def normalize(tensor, mean=[0.4914, 0.4822, 0.4465],
						std=[0.2023, 0.1994, 0.2010]):
	single_img = False
	if tensor.ndimension() == 3:
		single_img = True
		tensor = tensor[None,:,:,:]

	if not tensor.ndimension() == 4:
	    raise TypeError('tensor should be 4D')

	mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
	std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
	ret = tensor.sub(mean).div(std)
	return ret[0] if single_img else ret

def denormalize(tensor, mean=[0.4914, 0.4822, 0.4465],
						std=[0.2023, 0.1994, 0.2010]):
	single_img = False
	if tensor.ndimension() == 3:
		single_img = True
		tensor = tensor[None,:,:,:]

	if not tensor.ndimension() == 4:
	    raise TypeError('tensor should be 4D')

	mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
	std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
	ret = tensor.mul(std).add(mean)
	return ret[0] if single_img else ret
