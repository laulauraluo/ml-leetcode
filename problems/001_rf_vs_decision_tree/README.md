# 001 Random forest vs decision tree（sklearn）

用 sklearn 在同一份模拟数据上训练一棵 decision tree 和一个 random forest，比较两者的 test accuracy。

## 要求

写一个脚本，跑完后留下这些变量（测试会检查）：

| 变量 | 要求 |
|---|---|
| `X`, `y` | `make_classification` 生成：1000 个样本、4 个 feature、其中 2 个 informative，`random_state=42` |
| `X_train, X_test, y_train, y_test` | `train_test_split`，测试集占 20%，`random_state=42` |
| `dt_model`, `dt_score` | `DecisionTreeClassifier(random_state=42)` 在训练集上 fit，`dt_score` 是测试集 accuracy |
| `rf_model`, `rf_score` | `RandomForestClassifier`，100 棵树，`random_state=42`，同样 fit 再打分 |

最后把两个准确率 print 出来。

## 测试会检查

- 数据形状和划分
- 两个模型的类型、超参数，且都 fit 过
- `dt_score`、`rf_score` 和用同样设置重新算出来的一致
- 森林比单棵树准

## 做完再想

1. 把 `n_estimators` 改成 1，森林还比单棵树准吗？为什么？
2. 给决策树加上 `max_depth=3`，test acc 怎么变？这是在降 bias 还是降 variance？
3. 去 [002](../002_random_forest_from_scratch/README.md) 手写一遍，看看 `RandomForestClassifier` 里面到底做了什么。
