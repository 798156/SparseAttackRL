# sparse_attack_env_v2.py
"""
优化版本的稀疏攻击环境
主要改进：
1. 增强状态表示（加入梯度显著性图）
2. 改进的奖励函数（置信度感知）
3. 修改历史记录
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn.functional as F


class SparseAttackEnvV2(gym.Env):
    """
    优化版本的稀疏对抗攻击环境
    
    改进点：
    - 状态空间包含梯度显著性图
    - 细粒度的中间奖励
    - 置信度感知的奖励设计
    """

    def __init__(self, clean_image, true_label, model, max_steps=5, 
                 use_saliency=True, confidence_reward_weight=5.0):
        """
        初始化环境
        
        参数：
            clean_image: 原始图像 [C, H, W]
            true_label: 真实标签（整数）
            model: 目标模型（要攻击的对象）
            max_steps: 最大允许修改次数
            use_saliency: 是否使用显著性图增强状态
            confidence_reward_weight: 置信度奖励的权重
        """
        super(SparseAttackEnvV2, self).__init__()

        # 设备（GPU/CPU）
        self.device = next(model.parameters()).device
        self.model = model.to(self.device)

        # 攻击限制
        self.max_steps = max_steps
        self.current_step = 0
        
        # 配置
        self.use_saliency = use_saliency
        self.confidence_reward_weight = confidence_reward_weight

        # 原始图像增加 batch 维度 -> [1, C, H, W]
        self.clean_image = clean_image.unsqueeze(0).to(self.device)
        self.true_label = true_label
        
        # 当前对抗图像
        self.current_image = self.clean_image.clone()
        
        # 获取初始预测置信度
        with torch.no_grad():
            output = self.model(self.current_image)
            self.initial_confidence = torch.softmax(output, dim=1)[0, true_label].item()
            self.prev_confidence = self.initial_confidence

        # 获取图像尺寸
        _, C, H, W = self.current_image.shape
        self.height, self.width = H, W
        
        # 修改历史掩码
        self.modification_mask = torch.zeros((1, 1, H, W), device=self.device)

        # 动作空间：[x, y, dr, dg, db]
        self.action_space = spaces.Box(
            low=np.array([0, 0, -1.0, -1.0, -1.0]),
            high=np.array([W-1, H-1, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # 状态空间：增强版
        if use_saliency:
            # [图像(C) + 显著性图(1) + 置信度图(1) + 修改掩码(1)]
            state_channels = C + 3
        else:
            state_channels = C
            
        self.observation_space = spaces.Box(
            low=-5, high=5, shape=(state_channels, H, W), dtype=np.float32
        )

        # CIFAR-10 归一化参数
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(self.device)

    def _compute_saliency_map(self):
        """
        计算梯度显著性图
        指导智能体选择最有影响力的像素
        
        返回:
            saliency_map: [1, 1, H, W] 显著性图
        """
        # 需要梯度
        img = self.current_image.clone().detach().requires_grad_(True)
        
        # 前向传播
        output = self.model(img)
        
        # 计算目标类别的损失（我们想降低它）
        target_score = output[0, self.true_label]
        
        # 反向传播获取梯度
        self.model.zero_grad()
        target_score.backward()
        
        # 显著性 = 梯度的绝对值
        saliency = torch.abs(img.grad)
        
        # 聚合三个通道的显著性
        saliency_map = saliency.sum(dim=1, keepdim=True)  # [1, 1, H, W]
        
        # 归一化到 [0, 1]
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        
        return saliency_map

    def _compute_confidence_map(self):
        """
        计算当前预测的置信度分布图
        
        返回:
            confidence_map: [1, 1, H, W] 置信度图（当前为常数图）
        """
        with torch.no_grad():
            output = self.model(self.current_image)
            confidence = torch.softmax(output, dim=1)[0, self.true_label].item()
        
        # 创建常数置信度图
        confidence_map = torch.ones((1, 1, self.height, self.width), device=self.device) * confidence
        
        return confidence_map

    def _get_observation(self):
        """
        获取增强的观测状态
        
        返回:
            obs: 增强的状态 [C+3, H, W]
        """
        if not self.use_saliency:
            # 只返回图像
            return self.current_image.squeeze(0).cpu().detach().numpy()
        
        # 计算增强信息
        saliency_map = self._compute_saliency_map()
        confidence_map = self._compute_confidence_map()
        
        # 拼接所有通道
        # [图像(3) + 显著性(1) + 置信度(1) + 修改掩码(1)] = (6, H, W)
        enhanced_state = torch.cat([
            self.current_image,      # [1, 3, H, W]
            saliency_map,            # [1, 1, H, W]
            confidence_map,          # [1, 1, H, W]
            self.modification_mask   # [1, 1, H, W]
        ], dim=1)  # -> [1, 6, H, W]
        
        return enhanced_state.squeeze(0).cpu().detach().numpy()

    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # 重置状态
        self.current_image = self.clean_image.clone()
        self.current_step = 0
        self.modification_mask.zero_()
        
        # 重置置信度
        with torch.no_grad():
            output = self.model(self.current_image)
            self.prev_confidence = torch.softmax(output, dim=1)[0, self.true_label].item()

        info = {}
        return self._get_observation(), info

    def step(self, action):
        """
        执行一步动作
        
        改进的奖励函数：
        - 基础奖励（成功/失败/步骤）
        - 置信度变化奖励（中间指导）
        - 显著性奖励（鼓励修改重要像素）
        """
        x, y, dr, dg, db = action
        x = int(np.clip(x, 0, self.width - 1))
        y = int(np.clip(y, 0, self.height - 1))
        
        # 记录修改前的置信度
        prev_conf = self.prev_confidence

        # 修改像素
        img_unnorm = self.current_image * self.std + self.mean
        delta = torch.tensor([[dr, dg, db]]).view(1, 3, 1, 1).to(self.device) / 255.0
        img_unnorm[:, :, y:y + 1, x:x + 1] += delta
        img_unnorm = torch.clamp(img_unnorm, 0, 1)
        self.current_image = (img_unnorm - self.mean) / self.std
        
        # 更新修改掩码
        self.modification_mask[:, :, y, x] = 1.0
        
        # 模型推理
        with torch.no_grad():
            logits = self.model(self.current_image)
            pred_label = logits.argmax(dim=1).item()
            current_conf = torch.softmax(logits, dim=1)[0, self.true_label].item()

        # 判断攻击是否成功
        success = (pred_label != self.true_label)
        self.current_step += 1
        done = success or (self.current_step >= self.max_steps)

        # ========== 改进的奖励函数 ==========
        
        # 1. 基础终止奖励
        if success:
            reward = 10.0
        elif self.current_step >= self.max_steps:
            reward = -5.0
        else:
            reward = 0.0
        
        # 2. 置信度变化奖励（中间指导）⭐ 核心改进
        confidence_delta = prev_conf - current_conf  # 正值表示置信度下降（好事）
        reward += self.confidence_reward_weight * confidence_delta
        
        # 3. 步骤惩罚（鼓励稀疏性）
        reward -= 0.1
        
        # 4. 显著性奖励（可选，如果使用显著性图）
        if self.use_saliency and not success:
            # 如果修改了高显著性的像素，给予小奖励
            saliency_map = self._compute_saliency_map()
            pixel_saliency = saliency_map[0, 0, y, x].item()
            reward += 0.5 * pixel_saliency  # 鼓励修改重要像素
        
        # 更新置信度记录
        self.prev_confidence = current_conf

        # 信息记录
        info = {
            'success': success,
            'modified_pixel': (x, y),
            'current_pred': pred_label,
            'confidence': current_conf,
            'confidence_delta': confidence_delta,
            'step': self.current_step,
            'reward_breakdown': {
                'base': 10.0 if success else (-5.0 if done else 0.0),
                'confidence': self.confidence_reward_weight * confidence_delta,
                'step_penalty': -0.1,
                'saliency': 0.5 * pixel_saliency if (self.use_saliency and not success) else 0.0
            }
        }

        obs = self._get_observation()
        terminated = success
        truncated = (self.current_step >= self.max_steps) and not success

        return obs, reward, terminated, truncated, info

    def render(self):
        """显示图像（可选）"""
        pass


# 使用示例
if __name__ == "__main__":
    print("🧪 测试 SparseAttackEnvV2")
    
    from torchvision import datasets, transforms
    from target_model import load_target_model
    
    # 加载数据和模型
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    model = load_target_model('resnet18', num_classes=10)
    
    image, label = test_set[0]
    
    # 创建优化版环境
    env = SparseAttackEnvV2(image, label, model, max_steps=5, use_saliency=True)
    
    print(f"状态空间维度: {env.observation_space.shape}")
    print(f"动作空间维度: {env.action_space.shape}")
    
    # 测试一步
    obs, info = env.reset()
    print(f"\n初始状态形状: {obs.shape}")
    print(f"初始置信度: {info}")
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\n执行动作后:")
    print(f"奖励: {reward:.4f}")
    print(f"奖励分解: {info['reward_breakdown']}")
    print(f"置信度变化: {info['confidence_delta']:.4f}")
    
    print("\n✅ 测试完成！")

