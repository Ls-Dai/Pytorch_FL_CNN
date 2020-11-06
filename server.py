from models.cnn import Cnn
import torch

from utils import dir_setup
from configs import TrainConfig

config = TrainConfig()


class Server:
    def __init__(self, id_num):
        self.id = id_num

        self.local_dir = "severs/" + str(self.id) + "/"
        dir_setup(self.local_dir)

        self.model_dir = self.local_dir + "model/"
        dir_setup(self.model_dir)

        self.model = Cnn().to(config.device)

        self.model_name = "model.pkl"

    def save_model(self):
        torch.save(self.model, self.model_dir + self.model_name)

    def load_model(self):
        if config.no_cuda:
            self.model = torch.load(self.model_dir + self.model_name, map_location=torch.device("cpu"))
        else:
            self.model = torch.load(self.model_dir + self.model_name)