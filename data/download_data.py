import os
import requests
import numpy as np
import gzip

# 定义下载链接和数据保存路径
BASE_URL = "http://yann.lecun.com/exdb/mnist/"
DATA_PATH = "data/"
FILE_NAMES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
              "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]


def download_MNIST():
    """下载MNIST数据集"""
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    for file_name in FILE_NAMES:
        if not os.path.exists(os.path.join(DATA_PATH, file_name)):
            url = BASE_URL + file_name
            print(f"Downloading {url} ...")
            response = requests.get(url, stream=True)
            with open(os.path.join(DATA_PATH, file_name), "wb") as f:
                f.write(response.content)


def load_images(file_name):
    """从gzip文件中加载图像"""
    with gzip.open(file_name, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28)


def load_labels(file_name):
    """从gzip文件中加载标签"""
    with gzip.open(file_name, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)


def load_data():
    """加载MNIST数据集"""
    download_MNIST()

    x_train = load_images(os.path.join(DATA_PATH, FILE_NAMES[0]))
    y_train = load_labels(os.path.join(DATA_PATH, FILE_NAMES[1]))
    x_test = load_images(os.path.join(DATA_PATH, FILE_NAMES[2]))
    y_test = load_labels(os.path.join(DATA_PATH, FILE_NAMES[3]))

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_data()
    print("Data loaded successfully!")
