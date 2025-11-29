# 📅 今日任务 - Week 1 Day 2

**日期**：2025年11月4日  
**目标**：完成VGG16训练和完整实验

---

## 📋 任务清单

### 步骤1：训练VGG16模型（30-40分钟）

```bash
python train_cifar10_vgg16.py
```

**预期**：
- 训练30 epochs
- 最佳准确率：80-85%
- 模型保存为：`cifar10_vgg16.pth`

**注意事项**：
- VGG16参数比ResNet18多，训练稍慢
- 如果准确率<75%，可能需要调整训练参数
- 如果准确率>85%，那太棒了！

---

### 步骤2：验证VGG16准确率（3分钟）

训练完成后，验证模型：

```bash
python check_model_accuracy.py --model vgg16 --model_path cifar10_vgg16.pth
```

或者，可以修改`check_model_accuracy.py`来支持VGG16，或者直接在实验脚本中会自动验证。

---

### 步骤3：运行VGG16完整实验（2-3小时）

```bash
# 100样本测试
python run_vgg16_experiment.py --num_samples 100

# 如果时间充足，可以测试200样本
python run_vgg16_experiment.py --num_samples 200
```

**实验内容**：
- 测试5种攻击方法：
  - ✅ JSMA
  - ✅ One-Pixel
  - ✅ SparseFool
  - ⚠️ RL-V1（如果模型存在）
  - ⚠️ RL-V2（如果模型存在）

**预期结果**：
- ASR（攻击成功率）：40-80%
- L0（修改像素数）：1-10
- 结果保存在：`results/week1_day2/`

---

### 步骤4：对比ResNet18 vs VGG16（30分钟）

实验完成后，对比两个模型的结果：

```python
# 可以创建一个简单的对比脚本
python compare_resnet_vgg.py
```

或者手动对比：
1. 打开 `results/week1_day1/resnet18_summary.json`
2. 打开 `results/week1_day2/vgg16_summary.json`
3. 对比各个指标

**关注点**：
- 哪个模型更容易被攻击？
- 哪种攻击方法效果更好？
- L0有什么差异？

---

## 📊 预期产出

今天结束时应该有：

- ✅ VGG16模型（准确率80-85%）
- ✅ VGG16完整实验数据
- ✅ ResNet18 vs VGG16初步对比
- ✅ 发现的问题和观察

---

## ⚠️ 可能遇到的问题

### 问题1：VGG16训练很慢
**原因**：VGG16参数多（~138M vs ResNet18的~11M）  
**解决**：
- 减少batch size（如果显存不足）
- 或者耐心等待（30-40分钟）

### 问题2：RL方法报错
**原因**：RL模型是在ResNet18上训练的，可能不适配VGG16  
**解决**：
- 这是正常的！先跳过RL方法
- 只测试JSMA、One-Pixel、SparseFool
- RL方法需要重新训练（Week 3的任务）

### 问题3：准确率太低（<75%）
**原因**：训练不充分或参数不合适  
**解决**：
- 增加训练epochs（改为50）
- 或者降低学习率
- 或者使用更强的数据增强

---

## 🌙 今晚可做（可选）

### 1. 创建对比脚本（30分钟）

创建一个脚本来可视化对比ResNet18和VGG16：

```python
# compare_models.py
import json
import matplotlib.pyplot as plt

# 读取结果
with open('results/week1_day1/resnet18_summary.json') as f:
    resnet_results = json.load(f)

with open('results/week1_day2/vgg16_summary.json') as f:
    vgg_results = json.load(f)

# 绘制对比图
# ...
```

### 2. 分析初步观察（30分钟）

记录你的发现：
- VGG16和ResNet18哪个更鲁棒？
- 不同攻击方法的表现差异
- 有什么意外的发现？

### 3. 准备明天的任务（15分钟）

明天是Day 3：MobileNetV2实验
- 查看MobileNetV2的特点
- 思考可能的结果

---

## ✅ 今日成果检查

睡前检查：
- [ ] ✅ VGG16模型训练完成
- [ ] ✅ 模型准确率在75%以上
- [ ] ✅ VGG16实验运行完成
- [ ] ✅ 结果保存完整
- [ ] ✅ ResNet18 vs VGG16初步对比
- [ ] ✅ 记录观察和问题

---

## 🔔 明天预告 - Day 3

**主要任务**：MobileNetV2实验

**时间安排**：
- 训练MobileNetV2（30-40分钟）
- 运行实验（2-3小时）
- 三模型对比分析

---

## 💪 加油！

Day 2的任务比Day 1更顺利，因为：
- ✅ 已经有了完整的流程
- ✅ 脚本都准备好了
- ✅ 知道哪些坑要避免

**保持节奏，稳步推进！** 🎉

---

*生成时间：2025-11-04*








