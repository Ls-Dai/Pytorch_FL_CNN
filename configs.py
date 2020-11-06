import torch
import numpy as np


class TrainConfig():
    def __init__(self):
        self.num_of_clients = 4

        # train dataset setup
        self.batch_size = 50
        self.shuffle = False
        self.collate_fn = None
        self.batch_sampler = None
        self.sampler = None
        self.num_workers = 0
        self.pin_memory = False
        self.drop_last = False
        self.timeout = 0
        self.worker_init_fn = None
        self.order = True

        # train network setup
        self.epochs = 2000

        # CUDA setup
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # make use of GPU environment when existing
        self.no_cuda = not torch.cuda.is_available()

        self.valid_loss_min = np.Inf

        self.lr = 0.01
        self.momentum = 0.5
        self.seed = 1
        self.log_interval = 30
        self.save_model = True
        self.load_model = True

        # fed setup
        self.aggr_epochs = 200

        # model setup
        self.latent_dim = 100
        self.n_classes = 10
        self.img_size = 28
        self.channels = 1
        self.img_shape = (self.channels, self.img_size, self.img_size)

        # others
        self.train_data_path = r'datasets/train/'
        self.test_data_path = r'datasets/test/'
        self.load_model_path = r'savedmodels/'
        self.result_path = r'results/'

        self.save_model_name = r'mnist_cnn.pkl'
        self.load_model_name = r'mnist_cnn.pkl'


class TestConfig():
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 0
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 30
        self.save_model = True
        self.load_model = True

        self.train_data_path = r'datasets/train/'
        self.test_data_path = r'datasets/test/'
        self.load_model_path = r'savedmodels/'
        self.result_path = r'results/'

        self.save_model_name = r'mnist_cnn.pkl'
        self.load_model_name = r'mnist_cnn.pkl'
