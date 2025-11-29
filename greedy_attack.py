"""
Greedy Gradient Attack
简单但有效的稀疏攻击baseline方法
每次选择梯度最大的像素进行修改
"""

import torch
import numpy as np

def greedy_attack(image, label, model, max_pixels=10, step_size=0.3):
    """
    贪心梯度攻击
    
    原理：
    1. 计算损失关于输入的梯度
    2. 选择梯度绝对值最大的像素
    3. 沿负梯度方向修改
    4. 重复max_pixels次
    
    参数:
        image: 输入图像 (C, H, W)
        label: 真实标签
        model: 目标模型
        max_pixels: 最大修改像素数
        step_size: 每次修改的步长
    
    返回:
        success: 是否成功
        adv_image: 对抗样本
        modified_pixels: 修改的像素列表
    """
    device = next(model.parameters()).device
    adv_image = image.clone().to(device)
    
    # 归一化参数
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(device)
    
    modified_pixels = []
    modified_mask = torch.zeros_like(adv_image, dtype=torch.bool)
    
    # 首先检查初始预测
    with torch.no_grad():
        output = model(adv_image.unsqueeze(0))
        pred = output.argmax(dim=1).item()
        if pred != label:
            return False, adv_image, []
    
    # 迭代修改像素
    for step in range(max_pixels):
        # 检查是否已成功
        with torch.no_grad():
            output = model(adv_image.unsqueeze(0))
            pred = output.argmax(dim=1).item()
            if pred != label:
                return True, adv_image, modified_pixels
        
        # 计算梯度
        adv_image_batch = adv_image.unsqueeze(0).requires_grad_(True)
        output = model(adv_image_batch)
        
        # 对正确类别计算损失
        loss = output[0, label]
        
        model.zero_grad()
        loss.backward()
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
        
        # 修改像素
        with torch.no_grad():
            # 反归一化
            img_unnorm = adv_image * std + mean
            
            # 沿负梯度方向（降低正确类别的激活）
            perturbation = -torch.sign(grad[c, h, w]) * step_size
            img_unnorm[c, h, w] += perturbation
            
            # 裁剪到有效范围
            img_unnorm = torch.clamp(img_unnorm, 0, 1)
            
            # 重新归一化
            adv_image = (img_unnorm - mean) / std
        
        # 记录修改的像素
        modified_pixels.append((w, h, c))
        modified_mask[c, h, w] = True
    
    # 最终检查
    with torch.no_grad():
        output = model(adv_image.unsqueeze(0))
        pred = output.argmax(dim=1).item()
        success = (pred != label)
    
    return success, adv_image, modified_pixels


def greedy_attack_adaptive(image, label, model, max_pixels=10):
    """
    自适应步长的贪心攻击
    
    根据梯度大小自动调整步长
    """
    device = next(model.parameters()).device
    adv_image = image.clone().to(device)
    
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(device)
    
    modified_pixels = []
    modified_mask = torch.zeros_like(adv_image, dtype=torch.bool)
    
    # 检查初始预测
    with torch.no_grad():
        output = model(adv_image.unsqueeze(0))
        pred = output.argmax(dim=1).item()
        if pred != label:
            return False, adv_image, []
    
    for step in range(max_pixels):
        # 检查成功
        with torch.no_grad():
            output = model(adv_image.unsqueeze(0))
            pred = output.argmax(dim=1).item()
            if pred != label:
                return True, adv_image, modified_pixels
        
        # 计算梯度
        adv_image_batch = adv_image.unsqueeze(0).requires_grad_(True)
        output = model(adv_image_batch)
        loss = output[0, label]
        
        model.zero_grad()
        loss.backward()
        grad = adv_image_batch.grad[0]
        
        # 屏蔽已修改像素
        grad[modified_mask] = 0
        
        # 找到最大梯度
        grad_abs = torch.abs(grad)
        max_idx = grad_abs.argmax().item()
        
        C, H, W = adv_image.shape
        c = max_idx // (H * W)
        h = (max_idx % (H * W)) // W
        w = max_idx % W
        
        # 自适应步长：梯度越大，步长越大
        grad_magnitude = grad_abs[c, h, w].item()
        adaptive_step = min(0.5, 0.1 + grad_magnitude * 0.5)
        
        with torch.no_grad():
            img_unnorm = adv_image * std + mean
            perturbation = -torch.sign(grad[c, h, w]) * adaptive_step
            img_unnorm[c, h, w] += perturbation
            img_unnorm = torch.clamp(img_unnorm, 0, 1)
            adv_image = (img_unnorm - mean) / std
        
        modified_pixels.append((w, h, c))
        modified_mask[c, h, w] = True
    
    # 最终检查
    with torch.no_grad():
        output = model(adv_image.unsqueeze(0))
        pred = output.argmax(dim=1).item()
        success = (pred != label)
    
    return success, adv_image, modified_pixels


if __name__ == "__main__":
    print("🧪 测试 Greedy Gradient Attack")
    
    import torchvision
    import torchvision.transforms as transforms
    from torch import nn
    
    # 加载模型和数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load('cifar10_resnet18.pth', map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # 测试几个样本
    print("\n测试Greedy Attack:")
    successes = 0
    
    for i in range(10):
        image, label = testset[i]
        
        # 检查初始预测
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(device)).argmax(dim=1).item()
        
        if pred == label:
            success, adv_img, pixels = greedy_attack(
                image, label, model, max_pixels=10, step_size=0.3
            )
            
            if success:
                successes += 1
                print(f"样本 {i}: ✅ 成功 | 修改像素数: {len(pixels)}")
            else:
                print(f"样本 {i}: ❌ 失败")
    
    print(f"\n成功率: {successes}/10 = {successes/10*100:.1f}%")
    print("✅ 测试完成！")



