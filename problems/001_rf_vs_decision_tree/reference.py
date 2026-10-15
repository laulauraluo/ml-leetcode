from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. 生成一个简单的模拟数据集
X, y = make_classification(
    n_samples=1000, n_features=4, n_informative=2, random_state=42
)

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 训练单棵决策树（独裁专家）
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_score = dt_model.score(X_test, y_test)

# 4. 训练随机森林（专家团）
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_score = rf_model.score(X_test, y_test)

print(f"单决策树准确率: {dt_score:.4f}")
print(f"随机森林准确率: {rf_score:.4f}")
