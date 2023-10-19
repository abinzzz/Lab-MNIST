

# 简单图像分类器

本项目使用MNIST数据集实现了两种图像分类器：Softmax分类器和全连接神经网络分类器。

## 目录结构

```
.
|-- data/
|   |-- __init__.py
|   |-- download_data.py
|   |-- preprocess_data.py
|-- models/
|   |-- __init__.py
|   |-- softmax_classifier.py
|   |-- nn_classifier.py
|-- train.py
|-- predict.py
|-- README.md
```

## 环境要求

- Python 3.8+
- NumPy

## 如何使用

### 1. 下载数据

首先确保`download_data.py`在`data`目录中，并运行以下命令来下载数据：

```
python data/download_data.py
```

### 2. 训练模型

你可以通过运行以下命令来训练模型：

```
python train.py
```

这将训练一个全连接神经网络分类器。如果你想使用Softmax分类器，请在`train.py`中进行修改。

### 3. 预测

要对测试集进行预测，只需运行以下命令：

```
python predict.py
```

## 结构

- `data/download_data.py`：从MNIST数据源下载数据。
- `data/preprocess_data.py`：包含数据预处理函数，例如归一化和one-hot编码。
- `models/softmax_classifier.py`：Softmax分类器的实现。
- `models/nn_classifier.py`：全连接神经网络分类器的实现。

## 贡献

欢迎对项目进行改进和提交拉取请求！

## 许可

此项目采用MIT许可。

---

