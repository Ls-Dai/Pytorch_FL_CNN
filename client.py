import torch
import numpy as np
from torch.utils import data
from models.cnn import Cnn
import copy
from utils import *
import pandas as pd

from configs import TrainConfig


class Client:
    def __init__(self, id):
        self.id = id

        self.local_dir = 'clients/' + str(id) + '/'
        dir_setup(self.local_dir)

        self.dataset = []
        dir_setup(self.local_dir + 'dataset/')

        self.label = []
        dir_setup(self.local_dir + 'label/')

        self.model = Cnn()
        dir_setup(self.local_dir + 'model/')

        # The number of samples the client owns (before really load data)
        self.num_data_owned_setup = 0

        # log - train process visialization
        """if os.path.exists("clients/" + str(self.id) + "/" + "log.csv"):
            self.loss_list = []
        else:
            self.loss_list = []"""

    def load_data(self, data_label_list):
        self.dataset.append(data_label_list[0])
        self.label.append(data_label_list[1])

    def load_model_from_path(self, model_path):
        self.model = torch.load(model_path)

    def load_model(self, model):
        self.model = copy.deepcopy(model)

    def train_data_load(self, config: TrainConfig()):
        # Transform to torch tensors
        """dataset = []
        label = []
        for d in self.dataset:
            dataset.append(np.float(d))
        for d in self.label:
            label.append(np.float(d))"""
        '''for t in self.label:
            print(t)'''
        tensor_samples = torch.stack([s.float() for s in self.dataset])
        tensor_targets = torch.stack([t for t in self.label])

        train_dataset = data.TensorDataset(tensor_samples, tensor_targets)
        return data.DataLoader(dataset=train_dataset,
                               batch_size=config.batch_size,
                               shuffle=config.shuffle,
                               collate_fn=config.collate_fn,
                               batch_sampler=config.batch_sampler,
                               num_workers=config.num_workers,
                               pin_memory=config.pin_memory,
                               drop_last=config.drop_last,
                               timeout=config.timeout,
                               worker_init_fn=config.worker_init_fn)

    def num_data_owned(self):
        return len(self.dataset)

    # client writes logs
    def log_write(self, epoch, loss):
        loss_data_frame = pd.DataFrame(columns=None, index=[epoch], data=[loss])
        loss_data_frame.to_csv("clients/" + str(self.id) + "/" + "log.csv", mode='a', header=False)
