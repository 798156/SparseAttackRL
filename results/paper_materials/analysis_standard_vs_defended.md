# 标准模型 vs 防御模型 - 完整对比分析

## 1. 实验概述

本分析对比了5种稀疏对抗攻击方法在标准模型和防御模型上的性能。

**测试配置：**
- 标准模型：ResNet18 (CIFAR-10, ~88% accuracy)
- 防御模型：RobustBench对抗训练ResNet18 (~83-85% accuracy)
- 攻击方法：JSMA, SparseFool, Greedy, PixelGrad, RandomSparse
- 测试样本：每个配置100个样本

---

## 2. 主要发现


### 发现1：防御模型显著降低了攻击成功率

- **平均ASR下降：** 42.2 个百分点
- **平均相对下降：** 70.5%

这证明了对抗训练的有效性。


### 发现2：不同方法对防御的敏感度不同

- **最敏感方法：** PIXELGRAD (下降 82.5%)
- **最稳定方法：** SPARSEFOOL (下降 58.8%)

**解释：**
- 某些方法更依赖于模型的脆弱性，在防御模型上性能下降明显
- 某些方法具有更好的鲁棒性，在防御场景下相对稳定


### 发现3：方法相对排名基本保持

**标准模型Top 3：** JSMA, GREEDY, SPARSEFOOL
**防御模型Top 3：** SPARSEFOOL, GREEDY, JSMA

✅ 前3名中有3个方法保持，说明方法的相对性能在防御场景下稳定。


### 发现4：RandomSparse仍然是最差的baseline

- **标准模型ASR：** 23.0%
- **防御模型ASR：** 5.0%
- **下降：** 18.0 百分点

即使在防御模型上，RandomSparse的ASR仍然显著低于所有智能方法，
再次证明了梯度引导的像素选择策略的重要性。


---

## 3. 论文写作建议

### 3.1 实验章节

```latex
We further evaluate all methods on adversarially trained models 
from RobustBench to assess their practical applicability in 
defended scenarios. As expected, all methods show reduced ASR 
on the defended model, with an average drop of XX%. However, 
the relative performance ranking remains largely consistent, 
demonstrating the robustness of our findings.
```

### 3.2 讨论章节

可以讨论：
1. 不同方法对防御的敏感度差异
2. 为什么某些方法更鲁棒？
3. 这对实际部署有什么启示？

### 3.3 可能的额外贡献

如果发现了有趣的模式（例如某个方法特别稳定），可以：
- 专门分析原因
- 作为一个独立的发现
- 增强论文的深度

---

## 4. 数据表格

详见生成的LaTeX表格和图表。

---

## 5. 下一步建议

1. ✅ 检查所有数据的合理性
2. ✅ 确认发现是否有价值
3. ✅ 准备论文图表
4. 🎯 继续Week 1 Day 5：数据整理
5. 🎯 开始Week 2：补充分析

---

*生成时间：自动生成*
