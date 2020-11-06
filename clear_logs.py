import os
import re


def get_file_path(root_path_, file_list_, dir_list_):
    dir_or_files = os.listdir(root_path_)
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(root_path_, dir_file)
        if os.path.isdir(dir_file_path):
            dir_list_.append(dir_file_path)
            get_file_path(dir_file_path, file_list_, dir_list_)
        else:
            file_list_.append(dir_file_path)


if __name__ == "__main__":
    root_path = r"clients"
    file_list = []
    dir_list = []
    get_file_path(root_path, file_list, dir_list)
    # print(file_list)
    # print(dir_list)
    for file_path in file_list:
        if re.search(pattern='log.csv', string=file_path) is not None:
            os.remove(file_path)
    print("All log records are swiped!")