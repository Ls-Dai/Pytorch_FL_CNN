import os

def dir_setup(path):
    if not os.path.exists(path):
        os.makedirs(path)


"""def dir_setup(path):
    if not os.path.isdir(path):
        dir_setup(os.path.split(path)[0])
    else:
        return
    os.mkdir(path)"""
