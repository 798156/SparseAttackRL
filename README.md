# SparseAttackRL：基于强化学习的稀疏对抗攻击

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

本项目实现了使用**强化学习（Reinforcement Learning）**生成极低扰动的对抗样本，仅修改少量像素即可欺骗图像分类模型。该方法将稀疏对抗攻击建模为强化学习问题，使用PPO（Proximal Policy Optimization）算法训练智能体学习最优的攻击策略。

## 📋 目录

- [项目简介](#项目简介)
- [主要特性](#主要特性)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [技术细节](#技术细节)
- [配置说明](#配置说明)
- [实验结果](#实验结果)
- [使用示例](#使用示例)
- [常见问题](#常见问题)
- [参考文献](#参考文献)

## 🎯 项目简介

稀疏对抗攻击是生成对抗样本的一种特殊形式，其目标是使用**最少的像素修改**来欺骗深度学习模型。传统的启发式方法（如One-Pixel Attack）需要大量的随机搜索，效率较低。本项目创新性地将稀疏对抗攻击问题建模为强化学习环境，让智能体学习如何选择最优的像素位置和扰动值，从而在更短时间内生成更有效的对抗样本。

### 核心思想

- **状态空间**：当前对抗图像的像素值
- **动作空间**：像素坐标 (x, y) 和RGB扰动值 (dr, dg, db)
- **奖励机制**：攻击成功 +10.0，步骤惩罚 -0.1，失败惩罚 -5.0
- **目标**：最小化修改像素数量，最大化攻击成功率（ASR）

## ✨ 主要特性

- 🚀 **基于强化学习**：使用PPO算法训练智能体，自动学习最优攻击策略
- 🎯 **极低扰动**：仅需修改1-5个像素即可生成有效对抗样本
- 📊 **完整实验框架**：包含与One-Pixel Attack、JSMA Attack的对比实验
- 🧮 **多种攻击方法**：实现了SparseAttackRL、One-Pixel Attack、JSMA Attack三种方法
- 🔬 **多种模型支持**：支持ResNet18、VGG16、MobileNet等预训练模型
- 📈 **可视化分析**：自动生成ASR对比、像素分布、时间对比等图表
- ⚙️ **配置化管理**：通过YAML配置文件灵活调整实验参数
- 📝 **详细日志**：TensorBoard日志记录训练过程，便于分析和调试

## 📦 安装指南

### 环境要求

- Python 3.7+
- CUDA 11.0+ (可选，用于GPU加速)
- 8GB+ RAM
- 2GB+ 磁盘空间（用于CIFAR-10数据集）

### 安装步骤

1. **克隆项目**（如果是从GitHub获取）
```bash
git clone <repository-url>
cd SparseAttackRL
```

2. **安装依赖包**
```bash
pip install -r requirements.txt
```

### 依赖包说明

主要依赖包括：
- `torch>=1.13.0` - PyTorch深度学习框架
- `torchvision>=0.14.0` - 计算机视觉工具
- `gymnasium>=0.29.1` - 强化学习环境接口
- `stable-baselines3>=2.1.0` - PPO等RL算法实现
- `numpy>=1.21.0` - 数值计算
- `scipy>=1.7.0` - 科学计算（差分进化算法）
- `matplotlib>=3.5.0` - 可视化
- `PyYAML>=6.0` - 配置文件解析
- `tqdm>=4.64.0` - 进度条显示

## 🚀 快速开始

### 1. 基本使用

运行主程序进行单样本攻击：
```bash
python main.py
```

程序将自动：
1. 加载CIFAR-10数据集
2. 加载预训练的ResNet18模型
3. 训练PPO智能体（如果模型不存在）
4. 对测试样本执行攻击
5. 与One-Pixel Attack和JSMA Attack进行对比

### 2. 批量对比实验

程序默认会运行100个样本的对比实验，比较SparseAttackRL、One-Pixel Attack和JSMA Attack的性能。实验结束后会：
- 在控制台打印统计结果
- 将详细结果保存到 `results/experiment_*.txt`
- 生成可视化图表到 `results/plots/`

### 3. 单独测试JSMA攻击

如果只想测试JSMA攻击方法：
```bash
python test_jsma.py
```

### 4. 查看TensorBoard日志

训练过程中可以使用TensorBoard查看训练曲线：
```bash
tensorboard --logdir=./logs
```

然后在浏览器中打开 `http://localhost:6006`

## 📁 项目结构

```
SparseAttackRL/
├── main.py                    # 主程序入口，包含完整的对比实验框架
├── ppo_trainer.py             # PPO算法训练器
├── sparse_attack_env.py       # 强化学习环境（Gymnasium接口）
├── target_model.py            # 目标模型加载器（ResNet18/VGG16/MobileNet）
├── one_pixel_attack.py        # 对比算法：One-Pixel Attack（差分进化）
├── jsma_attack.py             # 对比算法：JSMA Attack（基于雅可比矩阵的显著性图攻击）
├── test_jsma.py               # JSMA攻击的独立测试脚本
├── visualization.py          # 可视化工具
├── config.yaml                # 配置文件
├── requirements.txt           # 依赖包列表
├── README.md                  # 项目文档（本文件）
├── data/                      # CIFAR-10数据集目录
│   └── cifar-10-batches-py/   # 数据集文件
├── logs/                      # TensorBoard日志目录
│   └── ppo_run_*/            # 各次训练的日志
├── results/                   # 实验结果目录
│   ├── experiment_*.txt      # 详细实验结果
│   └── plots/                # 可视化图表
│       ├── asr_comparison.png
│       ├── time_comparison.png
│       └── ...
└── ppo_sparse_model.zip      # 训练好的PPO模型（可选）
```

## 🔧 技术细节

### 强化学习环境设计

#### 状态空间（Observation Space）
- **维度**：`(C, H, W)` = `(3, 32, 32)` for CIFAR-10
- **内容**：当前对抗图像的归一化像素值
- **类型**：`Box(low=0, high=1, dtype=np.float32)`

#### 动作空间（Action Space）
- **维度**：5维向量 `[x, y, dr, dg, db]`
  - `x, y`: 要修改的像素坐标（0-31 for CIFAR-10）
  - `dr, dg, db`: RGB三个通道的扰动值（归一化到[-1, 1]）
- **类型**：`Box(low=[0, 0, -1, -1, -1], high=[W-1, H-1, 1, 1, 1])`

#### 奖励函数
```python
reward = {
    10.0   # 攻击成功（预测类别改变）
    -0.1   # 步骤惩罚（鼓励快速成功）
    -5.0   # 攻击失败（达到最大步数仍失败）
}
```

#### 终止条件
- **成功终止**：模型预测类别改变（`pred_label != true_label`）
- **失败终止**：达到最大修改步数（默认5步）

### PPO算法配置

```python
PPO(
    policy="MlpPolicy",        # MLP策略网络（处理展平的图像）
    learning_rate=3e-4,        # 学习率
    gamma=0.99,                # 折扣因子
    ent_coef=0.01,             # 熵系数（鼓励探索）
    batch_size=64,             # 批次大小
    device="auto"              # 自动选择CPU/GPU
)
```

### 像素修改机制

1. **反归一化**：将归一化的图像转换为原始像素空间 [0, 255]
2. **添加扰动**：在指定位置 (x, y) 的RGB通道上添加扰动值
3. **裁剪到有效范围**：`clamp(0, 1)` 确保像素值在有效范围内
4. **重新归一化**：转换回模型输入空间

### 对比算法：One-Pixel Attack

使用差分进化（Differential Evolution）算法：
- **优化变量**：`[x, y, r, g, b]`（坐标和RGB值）
- **优化目标**：最小化交叉熵损失，或直接优化误分类
- **迭代次数**：默认100次
- **种群大小**：10

### 对比算法：JSMA Attack

基于雅可比矩阵的显著性图攻击（Jacobian-based Saliency Map Attack）：
- **核心思想**：计算每个像素对分类结果的影响（雅可比矩阵），选择最有影响力的像素进行修改
- **显著性计算**：对于每个像素，计算其对目标类别和源类别激活的影响
  - 增加目标类别的激活：`∂F_target/∂x > 0`
  - 减少源类别的激活：`∂F_source/∂x < 0`
  - 显著性分数：`saliency = (∂F_target/∂x) × (-∂F_source/∂x)`
- **迭代过程**：
  1. 计算所有像素的显著性图
  2. 选择显著性最高的像素
  3. 修改该像素值（方向由梯度决定）
  4. 重复直到攻击成功或达到最大修改次数
- **参数设置**：
  - `max_pixels`: 最大修改像素数（默认5）
  - `theta`: 每次修改的幅度（默认1.0）
  - `targeted`: 是否为定向攻击（默认False）

## ⚙️ 配置说明

项目通过 `config.yaml` 文件进行配置，主要配置项如下：

```yaml
data:
  dataset: "CIFAR10"           # 数据集名称
  data_path: "./data"          # 数据路径
  batch_size: 1                # 批次大小

model:
  name: "resnet18"             # 模型名称：resnet18/vgg16/mobilenetv2
  pretrained: true             # 是否使用预训练权重
  num_classes: 10               # 分类类别数

attack:
  max_steps: 5                 # 最大修改像素数（稀疏度）
  epsilon: 16                   # 扰动强度上限（暂未使用）

rl:
  total_timesteps: 5000        # PPO训练总步数
  lr: 0.0003                    # 学习率
  gamma: 0.99                   # 折扣因子

use_gpu: true                  # 是否使用GPU
seed: 42                        # 随机种子（保证可复现性）
```

### 自定义配置

可以根据需要修改配置文件：
- **增加稀疏度**：增大 `attack.max_steps`（但可能降低稀疏性）
- **提高训练质量**：增大 `rl.total_timesteps`
- **更换目标模型**：修改 `model.name`
- **调整学习率**：修改 `rl.lr`

## 📊 实验结果

### 性能指标

项目在CIFAR-10测试集上进行了对比实验，主要评估指标包括：

1. **攻击成功率（ASR）**：成功欺骗模型的样本比例
2. **平均修改像素数**：成功攻击平均需要修改的像素数量
3. **平均攻击时间**：生成一个对抗样本所需的平均时间

### 典型结果示例

| 方法 | ASR (%) | 平均修改像素数 | 平均时间 (s) |
|------|---------|----------------|--------------|
| **SparseAttackRL** | ~XX.X | ~X.X | ~X.XX |
| **One-Pixel Attack** | ~XX.X | 1.0 | ~X.XX |
| **JSMA Attack** | ~XX.X | ~X.X | ~X.XX |

*注：具体数值取决于实验配置和随机种子，运行 `main.py` 会生成实际结果*

**三种方法的特点对比**：
- **SparseAttackRL**：通过强化学习自动学习攻击策略，平衡攻击成功率和修改像素数
- **One-Pixel Attack**：只修改1个像素，但需要较多迭代次数，攻击成功率可能较低
- **JSMA Attack**：基于梯度信息选择最有影响力的像素，攻击更具针对性，成功率较高

### 可视化结果

实验结果会自动保存到 `results/plots/` 目录，包括：
- `asr_comparison.png/pdf` - ASR对比柱状图（三种方法）
- `time_comparison.png/pdf` - 攻击时间对比（三种方法）
- `pixel_distribution.png/pdf` - 修改像素数分布（SparseAttackRL vs JSMA）

## 💡 使用示例

### 示例1：单样本攻击测试

```python
from target_model import load_target_model
from sparse_attack_env import SparseAttackEnv
from ppo_trainer import train_rl_agent
from stable_baselines3.common.vec_env import DummyVecEnv

# 1. 加载模型和数据
model = load_target_model("resnet18", num_classes=10)
image, label = test_set[0]  # 获取测试样本

# 2. 创建环境
env = SparseAttackEnv(image, label, model, max_steps=5)

# 3. 训练或加载智能体
agent = train_rl_agent(env, timesteps=5000)

# 4. 执行攻击
vec_env = DummyVecEnv([lambda: env])
obs = vec_env.reset()
done = False

while not done:
    action, _ = agent.predict(obs)
    obs, reward, terminated, truncated, info = vec_env.step(action)
    done = terminated[0] or truncated[0]
    
    if info[0]['success']:
        print(f"攻击成功！修改了 {info[0]['step']} 个像素")
        break
```

### 示例2：批量评估

```python
# 在 main.py 中已经实现了完整的批量评估
# 可以直接调用 run_full_comparison 函数

results = run_full_comparison(
    agent=agent,
    env_template=lambda img, lbl, mdl, steps: SparseAttackEnv(img, lbl, mdl, max_steps=steps),
    test_set=test_set,
    model=model,
    num_samples=100,
    max_rl_steps=5,
    one_pixel_max_iter=100
)
```

### 示例3：使用JSMA攻击

```python
from jsma_attack import jsma_attack
from target_model import load_target_model

# 加载模型和数据
model = load_target_model("resnet18", num_classes=10)
image, label = test_set[0]

# 执行JSMA攻击
success, adv_img, modified_pixels = jsma_attack(
    image=image,
    label=label,
    model=model,
    max_pixels=5,      # 最大修改5个像素
    theta=1.0,         # 每次修改幅度
    targeted=False     # 非定向攻击
)

if success:
    print(f"攻击成功！修改了 {len(modified_pixels)} 个像素")
    print(f"修改的像素位置: {modified_pixels}")
```

### 示例4：定向JSMA攻击

```python
# 定向攻击：将样本误分类为特定目标类别
target_class = 5  # 目标类别

success, adv_img, modified_pixels = jsma_attack(
    image=image,
    label=label,
    model=model,
    max_pixels=5,
    theta=1.0,
    targeted=True,           # 启用定向攻击
    target_class=target_class
)
```

### 示例5：自定义模型

```python
from target_model import load_target_model

# 支持多种模型
resnet = load_target_model("resnet18", num_classes=10)
vgg = load_target_model("vgg16", num_classes=10)
mobilenet = load_target_model("mobilenetv2", num_classes=10)
```

## ❓ 常见问题

### Q1: 训练需要多长时间？
**A:** 取决于硬件配置。在GPU上训练5000步大约需要5-10分钟，CPU上可能需要30-60分钟。

### Q2: 为什么攻击成功率不高？
**A:** 可能的原因：
- 训练步数不足（尝试增大 `total_timesteps`）
- 最大修改步数太少（尝试增大 `max_steps`）
- 目标模型过于鲁棒
- 数据集或模型加载有误

### Q3: 如何提高攻击成功率？
**A:** 
- 增加训练步数（`rl.total_timesteps`）
- 允许修改更多像素（`attack.max_steps`）
- 调整奖励函数权重
- 使用更强的预训练模型进行训练

### Q4: 支持其他数据集吗？
**A:** 当前代码主要针对CIFAR-10设计（32×32图像）。要支持其他数据集，需要：
- 修改 `sparse_attack_env.py` 中的图像尺寸
- 更新归一化参数（mean和std）
- 调整动作空间的范围

### Q5: 如何保存和加载训练好的模型？
**A:** 
```python
# 保存
agent.save("my_ppo_model")

# 加载
from stable_baselines3 import PPO
agent = PPO.load("my_ppo_model")
```

### Q6: GPU内存不足怎么办？
**A:** 
- 在 `config.yaml` 中设置 `use_gpu: false` 使用CPU
- 减小批次大小（修改PPO的 `batch_size`）
- 使用更小的模型（如MobileNet）

## 📚 参考文献

### 相关论文

1. **JSMA Attack** (2016)
   - Papernot, N., et al. "The limitations of deep learning in adversarial settings." *IEEE European symposium on security and privacy (EuroS&P)*, 2016.

2. **One-Pixel Attack** (2017)
   - Su, J., et al. "One pixel attack for fooling deep neural networks." *IEEE transactions on evolutionary computation*, 2019.

3. **Sparse Adversarial Attacks**
   - Modas, A., et al. "SparseFool: a few pixels make a big difference." *CVPR*, 2019.

4. **Reinforcement Learning for Adversarial Attacks**
   - Lin, Y., et al. "Nesterov accelerated gradient and scale invariance for adversarial attacks." *ICLR*, 2020.

5. **PPO Algorithm**
   - Schulman, J., et al. "Proximal policy optimization algorithms." *arXiv preprint arXiv:1707.06347*, 2017.

### 相关工具

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL算法库
- [Gymnasium](https://gymnasium.farama.org/) - RL环境接口
- [PyTorch](https://pytorch.org/) - 深度学习框架

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 👤 作者

- **项目名称**: SparseAttackRL
- **描述**: 基于强化学习的稀疏对抗攻击框架
- **创建时间**: 2024

## 🙏 致谢

- 感谢CIFAR-10数据集的提供者
- 感谢PyTorch和Stable-Baselines3社区的支持
- 感谢所有对抗攻击领域的研究者

---

**注意**：本项目仅用于学术研究和安全研究目的。请勿将对抗攻击技术用于任何恶意用途。
