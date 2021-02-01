# Pytorch_FL_CNN
An easy implementation of a federated learning scenario on CNN

## There are some discription about my code

### clear_logs.py
This will remove all the data of client folders, server folders and a log file in pwd. Make all records clear.

### client.py 
This is the module that contains a ```Client()``` class, which will be used as client module in FL. 

### configs.py
A module with a class that contains hyperparameters in both model part and traing part.

### init.py 
Module that would initialize a FL (assign servers and clients, and load dataset).

### pip_update_all.py
This will update all packages of pip.

### server.py 
The module that contains a ```Server()``` class, which will be used as server module in FL.

### test.py 
Deprecated. It was used to test a FL framework before.

### train.py 
Load configurations, clients, servers. Initialize a FL framework and train it.

### train_process_vis.py
This can read log files and visualize the training process after your training is completed.

### utils.py
Deprecated. All methods have been moved to client and server modules. 
