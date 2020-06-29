import torch
import torchvision

from .data_transforms import albumentations_transforms, torch_transforms
from utils import has_cuda, imshow

class DataEngine(object):

	classes = ["%s" % i for i in range(200)]

	def __init__(self, args):
		super(DataEngine, self).__init__()
		self.batch_size_cuda = args.batch_size_cuda
		self.batch_size_cpu = args.batch_size_cpu
		self.num_workers = args.num_workers
		self.train_data_path = args.train_data_path
		self.test_data_path = args.test_data_path
		self.load()

	def _transforms(self):
		# Data Transformations
		train_transform = albumentations_transforms(p=1.0, is_train=True)
		test_transform = albumentations_transforms(p=1.0, is_train=False)
		return train_transform, test_transform

	def _dataset(self):
		# Get data transforms
		train_transform, test_transform = self._transforms()

		# Dataset and Creating Train/Test Split
		train_set = torchvision.datasets.ImageFolder(root=self.train_data_path,
			transform=train_transform)
		test_set = torchvision.datasets.ImageFolder(root=self.test_data_path,
			transform=test_transform)
		return train_set, test_set

	def load(self):
		# Get Train and Test Data
		train_set, test_set = self._dataset()

		# Dataloader Arguments & Test/Train Dataloaders
		dataloader_args = dict(
			shuffle= True,
			batch_size= self.batch_size_cpu)
		if has_cuda():
			dataloader_args.update(
				batch_size= self.batch_size_cuda,
				num_workers= self.num_workers,
				pin_memory= True)

		self.train_loader = torch.utils.data.DataLoader(train_set, **dataloader_args)
		self.test_loader = torch.utils.data.DataLoader(test_set, **dataloader_args)

	def show_samples(self):
		# get some random training images
		dataiter = iter(self.train_loader)
		images, labels = dataiter.next()
		index = []
		num_img = min(len(self.classes), 10)
		for i in range(num_img):
			for j in range(len(labels)):
				if labels[j] == i:
					index.append(j)
					break
		if len(index) < num_img:
			for j in range(len(labels)):
				if len(index) == num_img:
					break
				if j not in index:
					index.append(j)
		imshow(torchvision.utils.make_grid(images[index],
				nrow=num_img, scale_each=True), "Sample train data")

