K-means 聚类算法是一种**无监督学习**方法，目标是把 n 个样本划分到 k 个“簇”（cluster），使得**同一个簇里的样本彼此相似**，不同簇的样本差异较大。  
其核心思想只有一句话：

> **“不断更新簇中心，再把样本重新分配给最近的中心。”**

---

### ✅ 算法步骤（ Lloyd 迭代 ）

1. **选中心**  
   随机选 k 个点作为初始簇中心 μ₁,…,μₖ。

2. **分配样本**  
   对每个样本 xᵢ，计算到各中心的欧氏距离，把它划入最近中心所在的簇：  
   cᵢ = argminⱼ ‖xᵢ − μⱼ‖²

3. **更新中心**  
   重新计算每个簇的均值作为新中心：  
   μⱼ = 1/|Cⱼ| · Σ_{x∈Cⱼ} x

4. **收敛判断**  
   重复 2-3 步，直到中心不再变化（或变化小于阈值）。

---

### ✅ 目标函数（最小化）

K-means 在最小化**簇内平方和**（Within-cluster Sum of Squares, WCSS）：

$$
J = \sum_{j=1}^{k} \sum_{x \in C_j} \|x - \mu_j\|^2
$$

---

### ✅ 优缺点速览

| 优点 | 缺点 |
|------|------|
| 简单、快速 | 需提前指定 k |
| 可解释性强 | 对初始中心敏感 |
| 适用于球形簇 | 对噪声、异常值敏感 |

---

### ✅ 小例子（NumPy 一手写）

```python
import numpy as np

def kmeans(X, k, max_iters=100):
    # 1. 随机初始化中心
    centers = X[np.random.choice(len(X), k, replace=False)]
    for _ in range(max_iters):
        # 2. 分配
        labels = np.argmin(np.linalg.norm(X[:, None] - centers, axis=2), axis=1)
        # 3. 更新
        new_centers = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return centers, labels
```

---

### ✅ 一句话总结

K-means = **“先定中心再分家，分完再把中心搬到家”** 的迭代过程。