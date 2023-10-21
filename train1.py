import numpy as np
import matplotlib.pyplot as plt
import pickle
from data.download_data import load_data
from data.preprocess_data import normalize, one_hot_encode
from models import NNClassifier
from models import SoftmaxClassifier
import time

# 参数设置
EPOCHS = 1000
LEARNING_RATE = 0.1 #0.07
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.1  # 10% 的训练数据用作验证

# 加载数据
(x_train, y_train), (x_test, y_test) = load_data()

# 数据预处理
x_train = normalize(x_train).reshape(x_train.shape[0], -1)
x_test = normalize(x_test).reshape(x_test.shape[0], -1)
y_train_encoded = one_hot_encode(y_train)

# 划分验证集
num_val_samples = int(VALIDATION_SPLIT * x_train.shape[0])
x_val = x_train[-num_val_samples:]
y_val = y_train_encoded[-num_val_samples:]
x_train = x_train[:-num_val_samples]
y_train_encoded = y_train_encoded[:-num_val_samples]

# 选择模型
model = NNClassifier(input_dim=x_train.shape[1], hidden_dim=100, num_classes=10)
#model = SoftmaxClassifier(input_dim=x_train.shape[1], num_classes=10)


# 训练模型并记录损失和准确率
train_losses = []
val_accuracies = []

start_time = time.time()

for epoch in range(EPOCHS):
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train_encoded = y_train_encoded[indices]

    epoch_losses = []
    for i in range(0, x_train.shape[0], BATCH_SIZE):
        x_batch = x_train[i:i + BATCH_SIZE]
        y_batch = y_train_encoded[i:i + BATCH_SIZE]
        loss = model.train(x_batch, np.argmax(y_batch, axis=1), LEARNING_RATE)
        epoch_losses.append(loss)

    train_losses.append(np.mean(epoch_losses))

    val_predictions = model.predict(x_val)
    val_accuracy = np.mean(val_predictions == y_train[-num_val_samples:])
    val_accuracies.append(val_accuracy)
    print(
        f"Epoch {epoch + 1}/{EPOCHS} - Training Loss: {train_losses[-1]:.4f} - Validation Accuracy: {val_accuracy * 100:.2f}%")


end_time = time.time()  # 结束计时

# 打印训练时间
training_time = end_time - start_time
print(f"Training took {training_time:.2f} seconds.")  # <-- 新增

# 可视化损失和准确率
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss over Epochs')

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Accuracy over Epochs')
plt.tight_layout()
plt.show()

# 保存模型权重
with open('model_weights.pkl', 'wb') as f:
    pickle.dump(model.get_weights(), f)

print("Training complete!")
