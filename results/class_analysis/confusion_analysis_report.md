# 混淆矩阵分析报告

**生成时间:** 2025-11-06 13:20:21  
**分析模型:** ResNet18  
**数据集:** CIFAR-10

---

## 1. 混淆矩阵概览

混淆矩阵展示了对抗攻击成功后，样本被误分类为哪个类别的分布。

### 1.1 生成的混淆矩阵


**JSMA:**
- 成功攻击总数: 23
- 可视化文件: `confusion_jsma.pdf`

**SPARSEFOOL:**
- 成功攻击总数: 17
- 可视化文件: `confusion_sparsefool.pdf`

**GREEDY:**
- 成功攻击总数: 12
- 可视化文件: `confusion_greedy.pdf`

**PIXELGRAD:**
- 成功攻击总数: 4
- 可视化文件: `confusion_pixelgrad.pdf`

**RANDOMSPARSE:**
- 成功攻击总数: 9
- 可视化文件: `confusion_randomsparse.pdf`

---

## 2. 主要混淆模式

### 2.1 JSMA

| 原始类别 | 最常误分类为 | 占比 |
|----------|-------------|------|
| automobile | airplane | 25.0% |
| bird | deer | 66.7% |
| deer | horse | 75.0% |
| dog | bird | 66.7% |
| frog | bird | 66.7% |
| horse | dog | 50.0% |
| ship | airplane | 50.0% |

### 2.2 SPARSEFOOL

| 原始类别 | 最常误分类为 | 占比 |
|----------|-------------|------|
| automobile | airplane | 33.3% |
| bird | deer | 66.7% |
| deer | horse | 100.0% |
| dog | bird | 50.0% |
| frog | bird | 66.7% |

### 2.3 GREEDY

| 原始类别 | 最常误分类为 | 占比 |
|----------|-------------|------|
| automobile | airplane | 50.0% |
| deer | horse | 100.0% |
| frog | bird | 66.7% |
| horse | dog | 50.0% |

### 2.4 PIXELGRAD

| 原始类别 | 最常误分类为 | 占比 |
|----------|-------------|------|

### 2.5 RANDOMSPARSE

| 原始类别 | 最常误分类为 | 占比 |
|----------|-------------|------|
| automobile | cat | 50.0% |
| frog | bird | 50.0% |

---

## 3. 跨方法对比

### 3.1 语义相似性混淆

分析是否存在跨方法一致的混淆模式（如猫↔狗）

| 类别对 | JSMA | SparseFool | Greedy | PixelGrad | RandomSparse |
|--------|------|------------|--------|-----------|---------------|
| Cat→Dog | 100.0% | 100.0% | 100.0% | - | 100.0% |
| Automobile→Truck | 25.0% | 33.3% | 0.0% | 0.0% | 50.0% |
| Bird→Frog | 0.0% | 0.0% | 0.0% | - | 0.0% |

---

## 4. 关键发现

1. **方法差异:** 不同方法的混淆模式有显著差异
2. **语义相似性:** 语义相似的类别更容易互相混淆
3. **攻击策略:** 某些方法倾向于跨越更远的类别


---

*报告生成时间: 2025-11-06 13:20:21*
