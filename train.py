import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils import data

from init import init_federated
from models.fed_merge import fedavg


def train(model, device, train_loader, optimizer, loss_func, epoch, client_id):
    model.train()
    train_loss = 0
    for batch_idx, (sample, target) in enumerate(train_loader):
        sample, target = sample.to(device), target.to(device)
        data_v = Variable(sample)
        target_v = Variable(target)
        optimizer.zero_grad()  # gradient = 0
        output = model(data_v)
        loss = loss_func(output, target_v)
        loss.backward()
        optimizer.step()  # gradient update
        """if (batch_idx + 1) % 10 == 0:
            # echo
            print('Client {}\tTrain Epoch: {} [{}/{} {:.0f}%]\tLoss: {:6f}'.format(client_id,
                                                                                   epoch,
                                                                                   batch_idx * len(sample),
                                                                                   len(train_loader.dataset),
                                                                                   100. * batch_idx / len(train_loader),
                                                                                   loss.data.item()
                                                                                   )
                  )"""
        train_loss = loss.data.item()

    return model.state_dict(), train_loss

def train_federated(config, clients, server):
    # print(server.model)
    print("###############")
    print("###############")
    print("###############")
    count = 1
    ###
    for epoch in range(1, config.epochs + 1):
        # A parameter collector
        para_collector = []

        # All clients update their local models
        for client in clients:
            optimizer = optim.Adam(client.model.parameters())
            loss_func = nn.CrossEntropyLoss()

            # This func would return the parameters of the model trained in this turn
            para, train_loss = train(model=client.model,
                         device=config.device,
                         train_loader=client.train_data_load(config),
                         optimizer=optimizer,
                         loss_func=loss_func,
                         epoch=epoch,
                         client_id=client.id)
            # echo
            print('Client {}\tTrain Epoch: {}\tLoss: {:6f}'.format(client.id, epoch, train_loss))

            # log write for this client
            client.log_write(epoch, train_loss)

            if epoch % config.aggr_epochs == 0:
                para_collector.append(copy.deepcopy(para))

        # federated!
        if epoch % config.aggr_epochs == 0:
            # merge + update global
            para_global = fedavg(para_collector)
            server.model.load_state_dict(para_global)
            # echo
            print("Server's model has been update, Fed No.: {}".format(count))
            count += 1

            # model download local
            for client in clients:
                client.load_model(copy.deepcopy(server.model))
                print("Client {}'s model has been updated from the server, Fed No.{}".format(client.id,
                                                                                             count))
    # Save the server model
    server.save_model()
    print("Global model has been saved on the server!")

    '''for client in clients:
        optimizer = optim.Adam(client.model.parameters())
        loss_func = nn.CrossEntropyLoss()

        for epoch in range(1, config.epochs + 1):
            train(model=client.model,
                  device=config.device,
                  train_loader=client.train_data_load(config),
                  optimizer=optimizer,
                  loss_func=loss_func,
                  epoch=epoch,
                  client_id=client.id)'''


if __name__ == '__main__':
    clients, server, config = init_federated()
    train_federated(config, clients, server)
