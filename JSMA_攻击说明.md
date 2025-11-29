# JSMA 攻击实现说明

## 📋 概述

本文档说明了新添加的 JSMA（Jacobian-based Saliency Map Attack）攻击方法的实现细节和使用方法。

## ✨ 新增内容

### 1. 核心文件

#### `jsma_attack.py`
实现了完整的JSMA攻击算法，包括：
- `jsma_attack()`: 主攻击函数
- `compute_jacobian()`: 计算雅可比矩阵
- `compute_saliency_map()`: 计算显著性图
- `jsma_attack_batch()`: 批量攻击接口

#### `test_jsma.py`
独立的JSMA攻击测试脚本，用于快速验证攻击效果。

### 2. 集成到主程序

修改了 `main.py`，将JSMA攻击集成到对比实验框架中：
- 单样本JSMA攻击测试（第98-107行）
- 批量对比实验（包含RL、One-Pixel、JSMA三种方法）
- 结果统计和可视化（三种方法的ASR、时间、像素分布对比）

### 3. 文档更新

更新了 `README.md`，添加了：
- JSMA攻击的技术细节说明
- 使用示例（非定向和定向攻击）
- 三种攻击方法的特点对比
- JSMA相关的参考文献

## 🔧 技术原理

### JSMA核心思想

JSMA通过计算每个像素对分类结果的影响来选择最有效的修改位置：

1. **雅可比矩阵计算**
   ```
   J[class_idx, c, h, w] = ∂F_class_idx / ∂x[c, h, w]
   ```
   其中 F_class_idx 是第 class_idx 类的输出激活

2. **显著性图计算**
   ```
   saliency = (∂F_target/∂x) × (-∂F_source/∂x)
   ```
   只有当以下条件同时满足时，像素才会被选中：
   - ∂F_target/∂x > 0 （增加像素值会增加目标类别激活）
   - ∂F_source/∂x < 0 （增加像素值会减少源类别激活）

3. **迭代修改**
   - 每次选择显著性最高的像素
   - 按梯度方向修改像素值
   - 重复直到攻击成功或达到最大修改次数

### 关键特性

- ✅ 支持非定向攻击（untargeted attack）
- ✅ 支持定向攻击（targeted attack）
- ✅ 自动处理归一化和反归一化
- ✅ 像素值裁剪到有效范围
- ✅ 跟踪已修改的像素位置

## 📝 使用方法

### 方法1：运行完整对比实验

```bash
python main.py
```

这会运行包含RL、One-Pixel和JSMA三种方法的完整对比实验。

### 方法2：单独测试JSMA

```bash
python test_jsma.py
```

快速测试5个样本的JSMA攻击效果。

### 方法3：在代码中使用

```python
from jsma_attack import jsma_attack
from target_model import load_target_model

# 加载模型
model = load_target_model("resnet18", num_classes=10)

# 执行攻击
success, adv_img, modified_pixels = jsma_attack(
    image=image,
    label=label,
    model=model,
    max_pixels=5,
    theta=1.0,
    targeted=False
)
```

## 🎯 参数说明

### `jsma_attack()` 函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image` | Tensor | - | 输入图像 (C, H, W) |
| `label` | int | - | 真实标签 |
| `model` | nn.Module | - | 目标模型 |
| `max_pixels` | int | 5 | 最大修改像素数 |
| `theta` | float | 1.0 | 每次修改的幅度 |
| `targeted` | bool | False | 是否为定向攻击 |
| `target_class` | int | None | 目标类别（仅当targeted=True时有效） |

### 返回值

- `success` (bool): 攻击是否成功
- `adv_image` (Tensor): 对抗样本
- `modified_pixels` (list): 修改的像素列表 [(x, y, channel), ...]

## 📊 实验结果

运行完整实验后，会生成以下输出：

### 控制台输出
```
📊 攻击性能对比:
----------------------------------------------------------------------
方法                           ASR (%)   平均修改像素数     平均时间 (s)
----------------------------------------------------------------------
SparseAttackRL (Ours)           XX.X           X.XX           X.XX
One-Pixel Attack                XX.X           1.00           X.XX
JSMA Attack                     XX.X           X.XX           X.XX
----------------------------------------------------------------------
```

### 文件输出
- `results/experiment_YYYYMMDD_HHMMSS.txt`: 详细实验结果
- `results/plots/asr_comparison.png/pdf`: ASR对比图
- `results/plots/time_comparison.png/pdf`: 时间对比图
- `results/plots/pixel_distribution.png/pdf`: 像素分布对比图

## 🔍 与其他方法的对比

### SparseAttackRL (强化学习方法)
- ✅ 优点：自适应学习，可能找到更优策略
- ❌ 缺点：需要训练时间，初期性能不稳定

### One-Pixel Attack (差分进化)
- ✅ 优点：只修改1个像素，极致的稀疏性
- ❌ 缺点：随机搜索效率低，成功率可能较低

### JSMA Attack (梯度引导)
- ✅ 优点：基于梯度信息，攻击更具针对性，成功率高
- ✅ 优点：计算效率相对较高
- ❌ 缺点：可能修改多个像素，依赖梯度计算

## ⚠️ 注意事项

1. **梯度计算开销**：JSMA需要对每个类别计算梯度，计算量较大
2. **归一化处理**：确保图像的归一化参数与模型训练时一致
3. **GPU内存**：计算雅可比矩阵时会占用较多GPU内存
4. **参数调整**：
   - `theta` 太大可能导致扰动过大
   - `max_pixels` 太小可能导致攻击失败率高

## 🐛 常见问题

### Q: JSMA攻击速度很慢？
A: JSMA需要计算雅可比矩阵，每次迭代都要对所有类别求梯度。可以尝试：
- 使用更小的模型
- 减少 `max_pixels` 参数
- 使用GPU加速

### Q: 攻击成功率不高？
A: 可以尝试：
- 增加 `max_pixels` 参数
- 调整 `theta` 参数
- 检查模型是否正确加载

### Q: 如何实现定向攻击？
A: 设置 `targeted=True` 和 `target_class`：
```python
success, adv_img, modified_pixels = jsma_attack(
    image=image,
    label=label,
    model=model,
    targeted=True,
    target_class=5  # 目标类别
)
```

## 📚 参考文献

Papernot, N., McDaniel, P., Jha, S., Fredrikson, M., Celik, Z. B., & Swami, A. (2016). 
"The limitations of deep learning in adversarial settings." 
*IEEE European symposium on security and privacy (EuroS&P)*, 372-387.

## 🎓 学术用途声明

本实现仅用于学术研究和安全研究目的。请勿将对抗攻击技术用于任何恶意用途。

---

**实现日期**: 2025年11月2日
**版本**: 1.0

