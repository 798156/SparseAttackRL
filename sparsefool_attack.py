# sparsefool_attack.py
"""
SparseFool攻击实现
参考：Modas et al., "SparseFool: a few pixels make a big difference", CVPR 2019

核心思想：迭代地找到最小L0扰动，每次修改对决策边界影响最大的像素
"""

import torch
import numpy as np


def sparsefool_attack(image, label, model, max_iterations=100, lambda_=3.0, overshoot=0.02):
    """
    SparseFool攻击
    
    参数:
        image: 输入图像 (C, H, W)
        label: 真实标签
        model: 目标模型
        max_iterations: 最大迭代次数
        lambda_: L0正则化参数（控制稀疏性）
        overshoot: 过冲参数
    
    返回:
        success: 是否成功
        adv_image: 对抗样本
        modified_pixels: 修改的像素列表
    """
    device = next(model.parameters()).device
    adv_image = image.clone().to(device).unsqueeze(0)
    
    # 记录修改的像素
    modified_pixels = []
    
    # 归一化参数
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(device)
    
    # 首先检查初始预测是否正确
    with torch.no_grad():
        initial_output = model(adv_image)
        initial_pred = initial_output.argmax(dim=1).item()
        
        # 如果初始预测就错了，跳过这个样本
        if initial_pred != label:
            return False, adv_image.squeeze(0), []
    
    for iteration in range(max_iterations):
        adv_image.requires_grad = True
        
        # 前向传播
        output = model(adv_image)
        pred = output.argmax(dim=1).item()
        
        # 检查是否成功
        if pred != label:
            return True, adv_image.squeeze(0).detach(), modified_pixels
        
        # 计算梯度
        model.zero_grad()
        output[0, label].backward()
        grad = adv_image.grad.data
        
        # 选择梯度最大的像素（对决策边界影响最大）
        grad_abs = torch.abs(grad[0])
        
        # 展平并找到最大梯度位置
        grad_flat = grad_abs.view(-1)
        max_idx = grad_flat.argmax().item()
        
        # 转换为坐标
        C, H, W = adv_image.shape[1:]
        c = max_idx // (H * W)
        h = (max_idx % (H * W)) // W
        w = max_idx % W
        
        # 计算扰动方向和大小
        # 简化版：沿负梯度方向移动
        perturbation_direction = -torch.sign(grad[0, c, h, w])
        
        # 应用扰动
        with torch.no_grad():
            # 反归一化
            img_unnorm = adv_image * std + mean
            
            # 添加扰动（小步长）
            step_size = 0.1 * (1 + overshoot)
            img_unnorm[0, c, h, w] += perturbation_direction * step_size
            
            # 裁剪到有效范围
            img_unnorm = torch.clamp(img_unnorm, 0, 1)
            
            # 重新归一化
            adv_image = (img_unnorm - mean) / std
            adv_image = adv_image.detach()
        
        # 记录修改的像素
        if (w, h, c) not in modified_pixels:
            modified_pixels.append((w, h, c))
    
    # 最后检查
    with torch.no_grad():
        output = model(adv_image)
        pred = output.argmax(dim=1).item()
        success = (pred != label)
    
    return success, adv_image.squeeze(0).detach(), modified_pixels


def sparsefool_attack_simple(image, label, model, max_pixels=5):
    """
    简化版SparseFool，限制最大修改像素数
    
    参数:
        image: 输入图像 (C, H, W)
        label: 真实标签
        model: 目标模型
        max_pixels: 最大修改像素数
    
    返回:
        success: 是否成功
        adv_image: 对抗样本
        modified_pixels: 修改的像素位置列表 [(x, y), ...]
    """
    device = next(model.parameters()).device
    adv_image = image.clone().to(device)
    
    # 归一化参数
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(device)
    
    modified_pixels = []
    modified_mask = torch.zeros_like(adv_image, dtype=torch.bool)
    
    # 首先检查初始预测是否正确
    with torch.no_grad():
        initial_output = model(adv_image.unsqueeze(0))
        initial_pred = initial_output.argmax(dim=1).item()
        
        # 如果初始预测就错了，跳过这个样本
        if initial_pred != label:
            return False, adv_image, []
    
    for step in range(max_pixels):
        # 检查是否成功
        with torch.no_grad():
            output = model(adv_image.unsqueeze(0))
            pred = output.argmax(dim=1).item()
            
            if pred != label:
                return True, adv_image, modified_pixels
        
        # 计算梯度
        adv_image_batch = adv_image.unsqueeze(0).requires_grad_(True)
        output = model(adv_image_batch)
        
        model.zero_grad()
        output[0, label].backward()
        grad = adv_image_batch.grad[0]
        
        # 屏蔽已修改的像素
        grad[modified_mask] = 0
        
        # 找到梯度绝对值最大的像素
        grad_abs = torch.abs(grad)
        max_idx = grad_abs.argmax().item()
        
        # 转换为坐标
        C, H, W = adv_image.shape
        c = max_idx // (H * W)
        h = (max_idx % (H * W)) // W
        w = max_idx % W
        
        # 应用扰动
        with torch.no_grad():
            # 反归一化
            img_unnorm = adv_image * std + mean
            
            # 沿负梯度方向扰动（降低正确类别的激活）
            # 增大扰动幅度以确保足够的影响
            perturbation = -torch.sign(grad[c, h, w]) * 0.8
            img_unnorm[c, h, w] += perturbation
            
            # 裁剪
            img_unnorm = torch.clamp(img_unnorm, 0, 1)
            
            # 重新归一化
            adv_image = (img_unnorm - mean) / std
        
        # 记录
        modified_pixels.append((w, h))
        modified_mask[c, h, w] = True
    
    # 最终检查
    with torch.no_grad():
        output = model(adv_image.unsqueeze(0))
        pred = output.argmax(dim=1).item()
        success = (pred != label)
    
    return success, adv_image, modified_pixels


# 使用示例和测试
if __name__ == "__main__":
    print("🧪 测试 SparseFool Attack")
    
    from torchvision import datasets, transforms
    from target_model import load_target_model
    
    # 加载数据和模型
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    model = load_target_model('resnet18', num_classes=10)
    
    # 测试几个样本
    print("\n测试 SparseFool Attack:")
    successes = 0
    
    for i in range(5):
        image, label = test_set[i]
        
        success, adv_img, pixels = sparsefool_attack_simple(
            image, label, model, max_pixels=5
        )
        
        if success:
            successes += 1
            print(f"样本 {i}: ✅ 成功 | 修改像素数: {len(pixels)}")
        else:
            print(f"样本 {i}: ❌ 失败")
    
    print(f"\n成功率: {successes}/5 = {successes/5*100:.1f}%")
    print("✅ 测试完成！")

