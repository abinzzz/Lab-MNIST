import numpy as np

def normalize(images):
    """对图像数据进行归一化"""
    return (images.astype(np.float32) - 127.5) / 127.5

def one_hot_encode(labels, num_classes=10):
    """对标签进行独热编码"""
    encoded = np.zeros((labels.size, num_classes), dtype=np.float32)
    for idx, label in enumerate(labels):
        encoded[idx, label] = 1.0
    return encoded

if __name__ == "__main__":
    # 作为一个例子
    from download_data import load_data

    (x_train, y_train), (x_test, y_test) = load_data()

    x_train_normalized = normalize(x_train)
    y_train_encoded = one_hot_encode(y_train)

    print("Data preprocessing done!")
