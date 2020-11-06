import os

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms

from client import Client
from server import Server
from configs import TrainConfig


def load_datasets(clients, config):
    train_dataset = datasets.MNIST('datasets/', download=True, train=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,)),
                                   ]))

    num_data_owned_setup = 12500

    train_distributer = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=num_data_owned_setup * len(clients),
                                                    shuffle=config.shuffle)

    images, labels = next(iter(train_distributer))
    
    # Hetero

    if config.order:
        images_array = images.numpy()
        labels_array = labels.numpy()
        images_list = []
        labels_list = []
        for i in range(config.n_classes):
            for j in range(num_data_owned_setup * len(clients)):
                if labels_array[j] == i:
                    images_list.append(images_array[j])
                    labels_list.append(labels_array[j])
        images_array = np.array(images_list)
        labels_array = np.array(labels_list)
        images = torch.from_numpy(images_array)
        labels = torch.from_numpy(labels_array)


    else:
        images, labels = next(iter(train_distributer))

    for client in clients:
        for i in range(num_data_owned_setup):
            j = i + client.id * num_data_owned_setup
            client.load_data([images[j], labels[j]])


def init_federated():
    # clients list
    clients = []

    # load configs
    config = TrainConfig()

    # generate clients
    for i in range(config.num_of_clients):
        clients.append(Client(i))

    # generate server
    server = Server(0)
    if os.path.exists(server.model_dir + server.model_name):
        server.load_model()
        print("Global model saved on the server has been restored!")
    else:
        print("Global model has been created!")
    # load datasets
    load_datasets(clients=clients, config=config)

    # load models
    for client in clients:
        client.load_model(server.model)

    return clients, server, config


if __name__ == '__main__':
    clients, server, config = init_federated()
    # print(clients)
    # print(server)
    # print(config)

    #### Test Code ####

    # print(np.array(clients[0].dataset[0][0]))

    pic = np.array([])
    for i in range(50):
        pic_h = np.array([])
        for j in range(50):
            if j == 0:
                pic_h = np.array(clients[0].dataset[i * 50 + j][0])
            else:
                # print(pic_h)
                # print(np.array(clients[0].dataset[i * 50 + j][0]))
                pic_h = np.hstack((pic_h, np.array(clients[0].dataset[i * 50 + j][0])))
        if i == 0:
            pic = pic_h
        else:
            pic = np.vstack((pic, pic_h))

    cv2.imshow("pic", pic)
    cv2.waitKey(0)

    #### Test Code ####

    shutil.rmtree("clients")
    shutil.rmtree("severs")
