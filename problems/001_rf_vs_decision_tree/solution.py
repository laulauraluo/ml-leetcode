from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. 生成模拟数据：1000 个样本、4 个 feature（其中 2 个 informative），random_state=42
X, y = None, None

# 2. 划分训练集和测试集：测试集占 20%，random_state=42
X_train, X_test, y_train, y_test = None, None, None, None

# 3. 训练单棵决策树（random_state=42），在测试集上算准确率
dt_model = None
dt_score = None

# 4. 训练随机森林（100 棵树，random_state=42），在测试集上算准确率
rf_model = None
rf_score = None

print(f"单决策树准确率: {dt_score}")
print(f"随机森林准确率: {rf_score}")
