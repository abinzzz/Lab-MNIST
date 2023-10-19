import numpy as np
from data.download_data import load_data
from data.preprocess_data import normalize
from models import NNClassifier
from models import SoftmaxClassifier
import pickle
# 加载数据（这里只加载测试集）
(_, _), (x_test, y_test) = load_data()

# 数据预处理
x_test = normalize(x_test).reshape(x_test.shape[0], -1)

# 加载模型
# 注意：这里的加载只是初始化。在实际应用中，我们可能还需要一个步骤来加载模型权重。
# 例如：model.load_weights('path_to_weights.pkl')
model = NNClassifier(input_dim=x_test.shape[1], hidden_dim=100, num_classes=10)
#model = SoftmaxClassifier(input_dim=x_test.shape[1], num_classes=10)

# 加载权重
with open('model_weights.pkl', 'rb') as f:
    weights = pickle.load(f)
model.set_weights(weights)

# 进行预测
predictions = model.predict(x_test)

# 输出预测结果或评估性能
accuracy = np.mean(predictions == y_test)
print(f"Accuracy on test set: {accuracy * 100:.2f}%")

