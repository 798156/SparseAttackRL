import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

class SparseAttackEnv(gym.Env):
    """
    将稀疏对抗攻击建模为强化学习环境。
    智能体每次选择一个像素 (x,y) 并施加 RGB 扰动，目标是以最少修改次数使模型误分类。
    """

    def __init__(self, clean_image, true_label, model, max_steps=5):
        """
        初始化环境
        参数：
            clean_image: 原始图像 [C, H, W]
            true_label: 真实标签（整数）
            model: 目标模型（要攻击的对象）
            max_steps: 最大允许修改次数（决定稀疏程度）
        """
        super(SparseAttackEnv, self).__init__()

        # 设备（GPU/CPU）
        self.device = next(model.parameters()).device
        self.model = model.to(self.device)

        # 攻击限制
        self.max_steps = max_steps
        self.current_step = 0  # 当前已修改几步

        # 原始图像增加 batch 维度 -> [1, C, H, W]
        self.clean_image = clean_image.unsqueeze(0).to(self.device)
        self.true_label = true_label
        # 当前对抗图像（初始等于原始图）
        self.current_image = self.clean_image.clone().requires_grad_(True)

        # 获取图像尺寸
        _, C, H, W = self.current_image.shape
        self.height, self.width = H, W

        # ======================
        # 动作空间定义
        # ======================
        # 动作是一个五维向量：[x, y, dr, dg, db]
        #   x, y: 要修改的像素坐标
        #   dr, dg, db: R/G/B 三个通道的扰动值（归一化到 [-1,1]）
        self.action_space = spaces.Box(
            low=np.array([0, 0, -1.0, -1.0, -1.0]),    # 最小值
            high=np.array([W-1, H-1, 1.0, 1.0, 1.0]),   # 最大值
            dtype=np.float32  # 数据类型
        )

        # ======================
        # 状态空间定义
        # ======================
        # 状态就是当前的对抗图像本身 [C, H, W]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(C, H, W), dtype=np.float32
        )

        # CIFAR-10 归一化参数（必须反归一化才能正确修改像素）
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(self.device)

    def reset(self, seed=None, options=None):
        """
        重置环境
        """
        # 设置随机种子（可选）
        if seed is not None:
            import numpy as np
            np.random.seed(seed)
            torch.manual_seed(seed)

        # 重置状态
        self.current_image = self.clean_image.clone().requires_grad_(True)
        self.current_step = 0

        # 返回观测值和 info 字典（新 API 要求）
        info = {}  # 可添加额外信息
        return self.current_image.squeeze(0).cpu().detach().numpy(), info
        #                                 ↑↑↑↑↑↑↑↑ 加上 .detach()

    def step(self, action):
        """
        执行一步动作
        """
        x, y, dr, dg, db = action
        x = int(np.clip(x, 0, self.width - 1))
        y = int(np.clip(y, 0, self.height - 1))

        # -------------------------------
        # 修改像素（已修复维度问题）
        # -------------------------------
        img_unnorm = self.current_image * self.std + self.mean
        delta = torch.tensor([[dr, dg, db]]).view(1, 3, 1, 1).to(self.device) / 255.0
        img_unnorm[:, :, y:y + 1, x:x + 1] += delta
        img_unnorm = torch.clamp(img_unnorm, 0, 1)
        self.current_image = (img_unnorm - self.mean) / self.std

        # -------------------------------
        # 模型推理
        # -------------------------------
        with torch.no_grad():
            logits = self.model(self.current_image)
            pred_label = logits.argmax(dim=1).item()  # 返回 Python int
            confidence = torch.softmax(logits, dim=1)[0, pred_label].item()

        # -------------------------------
        # ✅ 修复这里：去掉 .item()
        # -------------------------------
        success = (pred_label != self.true_label)  # 两个都是 int，直接比较

        done = success or (self.current_step >= self.max_steps)
        reward = 10.0 if success else (-0.1 if not done else -5.0)
        self.current_step += 1

        info = {
            'success': success,
            'modified_pixel': (x, y),
            'current_pred': pred_label,
            'confidence': confidence,
            'step': self.current_step
        }

        # 返回五元组
        obs = self.current_image.squeeze(0).cpu().detach().numpy()
        terminated = success
        truncated = (self.current_step >= self.max_steps) and not success

        return obs, reward, terminated, truncated, info

    def render(self):
        """
        显示图像（可选）
        """
        pass