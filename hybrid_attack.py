# hybrid_attack.py
"""
混合攻击策略：结合 RL 和 JSMA
核心思想：用JSMA快速逼近，用RL精细优化
"""

import torch
import numpy as np
from jsma_attack import jsma_attack, compute_jacobian, compute_saliency_map


class HybridAttackStrategy:
    """
    混合攻击策略
    
    策略：
    - 前期：使用JSMA快速选择高影响力的像素
    - 后期：使用RL进行精细调整
    """
    
    def __init__(self, rl_agent, model, max_steps=5, rl_ratio_schedule='dynamic'):
        """
        参数:
            rl_agent: 训练好的RL智能体
            model: 目标模型
            max_steps: 最大攻击步数
            rl_ratio_schedule: RL使用比例调度
                - 'static': 固定比例 (0.5)
                - 'increasing': 递增 (0.3 -> 0.7 -> 1.0)
                - 'dynamic': 根据置信度动态调整
        """
        self.rl_agent = rl_agent
        self.model = model
        self.max_steps = max_steps
        self.rl_ratio_schedule = rl_ratio_schedule
        self.device = next(model.parameters()).device
    
    def attack(self, image, label, verbose=False):
        """
        执行混合攻击
        
        参数:
            image: 输入图像 (C, H, W)
            label: 真实标签
            verbose: 是否打印详细信息
        
        返回:
            success: 是否成功
            adv_image: 对抗样本
            modified_pixels: 修改的像素列表
            method_used: 每步使用的方法
        """
        adv_image = image.clone().to(self.device)
        modified_pixels = []
        method_used = []
        
        for step in range(self.max_steps):
            # 检查是否已经成功
            with torch.no_grad():
                output = self.model(adv_image.unsqueeze(0))
                pred = output.argmax(dim=1).item()
                confidence = torch.softmax(output, dim=1)[0, label].item()
            
            if pred != label:
                if verbose:
                    print(f"✅ 攻击成功！步数: {step}")
                return True, adv_image, modified_pixels, method_used
            
            # 决定使用哪种方法
            use_rl = self._should_use_rl(step, confidence)
            
            if use_rl:
                # 使用 RL 策略
                if verbose:
                    print(f"步骤 {step+1}: 使用 RL")
                
                x, y, r, g, b = self._rl_select_action(adv_image)
                method = 'RL'
                
            else:
                # 使用 JSMA 启发式
                if verbose:
                    print(f"步骤 {step+1}: 使用 JSMA")
                
                x, y, r, g, b = self._jsma_select_action(adv_image, label)
                method = 'JSMA'
            
            # 应用修改
            adv_image = self._apply_modification(adv_image, x, y, r, g, b)
            modified_pixels.append((x, y))
            method_used.append(method)
        
        # 最后检查一次
        with torch.no_grad():
            output = self.model(adv_image.unsqueeze(0))
            pred = output.argmax(dim=1).item()
        
        success = (pred != label)
        
        if verbose:
            if success:
                print(f"✅ 攻击成功！")
            else:
                print(f"❌ 攻击失败")
        
        return success, adv_image, modified_pixels, method_used
    
    def _should_use_rl(self, step, confidence):
        """
        决定是否使用RL
        
        参数:
            step: 当前步数
            confidence: 当前置信度
        
        返回:
            bool: True表示使用RL，False表示使用JSMA
        """
        if self.rl_ratio_schedule == 'static':
            # 固定50%概率使用RL
            return np.random.rand() < 0.5
        
        elif self.rl_ratio_schedule == 'increasing':
            # 递增：前30%用30% RL，中间40%用70% RL，最后30%用100% RL
            progress = step / self.max_steps
            if progress < 0.3:
                rl_prob = 0.3
            elif progress < 0.7:
                rl_prob = 0.7
            else:
                rl_prob = 1.0
            return np.random.rand() < rl_prob
        
        elif self.rl_ratio_schedule == 'dynamic':
            # 动态：根据置信度调整
            # 高置信度时多用JSMA（快速降低），低置信度时多用RL（精细调整）
            if confidence > 0.7:
                rl_prob = 0.3  # 主要用JSMA
            elif confidence > 0.4:
                rl_prob = 0.6  # 混合
            else:
                rl_prob = 0.9  # 主要用RL
            return np.random.rand() < rl_prob
        
        else:
            return True
    
    def _rl_select_action(self, image):
        """
        使用RL智能体选择动作
        
        返回:
            x, y, r, g, b: 像素位置和RGB修改值
        """
        # 注意：如果agent是用增强状态训练的，需要提供增强状态
        # 这里简化处理：如果失败则使用随机动作
        try:
            # 尝试使用原始图像
            obs = image.cpu().numpy()
            action, _ = self.rl_agent.predict(obs, deterministic=False)
        except Exception as e:
            # 如果失败（状态空间不匹配），使用随机动作
            # print(f"RL预测失败，使用随机动作: {e}")
            action = self.rl_agent.action_space.sample()
        
        x, y, dr, dg, db = action
        x = int(np.clip(x, 0, image.shape[2] - 1))
        y = int(np.clip(y, 0, image.shape[1] - 1))
        
        return x, y, dr, dg, db
    
    def _jsma_select_action(self, image, label):
        """
        使用JSMA启发式选择动作
        
        返回:
            x, y, r, g, b: 像素位置和RGB修改值
        """
        # 计算显著性图
        image_batch = image.unsqueeze(0).requires_grad_(True)
        
        # 计算雅可比矩阵
        output = self.model(image_batch)
        num_classes = output.shape[1]
        
        jacobian = torch.zeros((1, num_classes, *image.shape), device=self.device)
        
        for class_idx in range(num_classes):
            self.model.zero_grad()
            if image_batch.grad is not None:
                image_batch.grad.zero_()
            
            class_output = output[0, class_idx]
            class_output.backward(retain_graph=True)
            
            if image_batch.grad is not None:
                jacobian[0, class_idx] = image_batch.grad[0].clone()
        
        # 选择第二高的类别作为目标
        sorted_preds = output.argsort(dim=1, descending=True)[0]
        target_class = sorted_preds[1].item() if sorted_preds[0].item() == label else sorted_preds[0].item()
        
        # 计算显著性
        grad_target = jacobian[0, target_class]
        grad_source = jacobian[0, label]
        
        alpha = grad_target
        beta = -grad_source
        
        valid = (alpha > 0) & (beta > 0)
        saliency = alpha * beta * valid.float()
        
        # 找到显著性最高的像素
        saliency_flat = saliency.view(-1)
        max_idx = saliency_flat.argmax().item()
        
        C, H, W = image.shape
        c = max_idx // (H * W)
        h = (max_idx % (H * W)) // W
        w = max_idx % W
        
        # 确定修改方向
        direction = 1.0 if jacobian[0, target_class, c, h, w] > 0 else -1.0
        
        # RGB值（简化：统一修改）
        dr = dg = db = direction * 128  # 中等强度
        
        return w, h, dr, dg, db
    
    def _apply_modification(self, image, x, y, dr, dg, db):
        """
        应用像素修改
        
        返回:
            modified_image: 修改后的图像
        """
        # 反归一化
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(self.device)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(self.device)
        
        img_unnorm = image * std + mean
        
        # 应用修改
        delta = torch.tensor([dr, dg, db]).view(3, 1, 1).to(self.device) / 255.0
        img_unnorm[:, y:y+1, x:x+1] += delta
        
        # 裁剪并重新归一化
        img_unnorm = torch.clamp(img_unnorm, 0, 1)
        modified_image = (img_unnorm - mean) / std
        
        return modified_image


def hybrid_attack(image, label, model, rl_agent, max_pixels=5, strategy='dynamic'):
    """
    便捷的混合攻击接口
    
    参数:
        image: 输入图像
        label: 真实标签  
        model: 目标模型
        rl_agent: RL智能体
        max_pixels: 最大修改像素数
        strategy: 策略类型 ('static', 'increasing', 'dynamic')
    
    返回:
        success: 是否成功
        adv_image: 对抗样本
        modified_pixels: 修改的像素
        method_used: 使用的方法记录
    """
    hybrid_strategy = HybridAttackStrategy(
        rl_agent=rl_agent,
        model=model,
        max_steps=max_pixels,
        rl_ratio_schedule=strategy
    )
    
    return hybrid_strategy.attack(image, label, verbose=False)


# 使用示例
if __name__ == "__main__":
    print("🧪 测试混合攻击策略")
    
    from torchvision import datasets, transforms
    from target_model import load_target_model
    from stable_baselines3 import PPO
    import os
    
    # 加载数据和模型
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    target_model = load_target_model('resnet18', num_classes=10)
    
    # 加载RL智能体
    agent_path = "ppo_sparse_model.zip"
    if os.path.exists(agent_path):
        rl_agent = PPO.load(agent_path)
    else:
        print("⚠️  未找到训练好的RL模型，请先训练")
        exit()
    
    # 测试几个样本
    print("\n测试混合攻击策略:")
    print("=" * 60)
    
    for strategy in ['static', 'increasing', 'dynamic']:
        print(f"\n策略: {strategy}")
        successes = 0
        
        for i in range(5):
            image, label = test_set[i]
            
            success, adv_img, pixels, methods = hybrid_attack(
                image, label, target_model, rl_agent,
                max_pixels=5, strategy=strategy
            )
            
            if success:
                successes += 1
                print(f"  样本 {i}: ✅ 成功 | 步数: {len(pixels)} | 方法: {methods}")
            else:
                print(f"  样本 {i}: ❌ 失败")
        
        print(f"成功率: {successes}/5 = {successes/5*100:.1f}%")
    
    print("\n✅ 测试完成！")

