import pprint

class ModelConfig(object):

	def __init__(self,):
		super(ModelConfig, self).__init__()
		self.seed = 1
		self.batch_size_cuda = 256
		self.batch_size_cpu = 128
		self.num_workers = 4
		# Regularization
		self.dropout = 0
		self.l1_decay = 0
		self.l2_decay = 5e-3
		self.lr = 0.001
		self.momentum = 0.9
		self.epochs = 15
		self.train_data_path = "/content/data/tiny-imagenet-200/new_train"
		self.test_data_path = "/content/data/tiny-imagenet-200/new_test"

	def print_config(self):
		print("Model Parameters:")
		pprint.pprint(vars(self), indent=2)

def test_config():
	args = ModelConfig()
	args.print_config()

if __name__ == '__main__':
	test_config()
