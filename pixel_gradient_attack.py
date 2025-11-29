"""
Pixel Gradient Attack (改进的梯度攻击)
与Greedy不同，这个方法使用累积梯度和自适应步长
"""

import torch
import numpy as np

def pixel_gradient_attack(image, label, model, max_pixels=10, alpha=0.2, beta=0.9):
    """
    基于像素梯度的改进攻击
    
    原理：
    1. 维护每个像素的累积梯度重要性
    2. 使用动量更新（类似Adam）
    3. 自适应选择最优像素
    
    参数:
        image: 输入图像 (C, H, W)
        label: 真实标签
        model: 目标模型
        max_pixels: 最大修改像素数
        alpha: 更新步长
        beta: 动量系数
    
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
    
    # 累积梯度（动量）
    momentum = torch.zeros_like(adv_image)
    
    # 首先检查初始预测
    with torch.no_grad():
        output = model(adv_image.unsqueeze(0))
        pred = output.argmax(dim=1).item()
        if pred != label:
            return False, adv_image, []
    
    C, H, W = adv_image.shape
    
    # 迭代修改像素
    for step in range(max_pixels):
        # 提前检查是否成功
        with torch.no_grad():
            output = model(adv_image.unsqueeze(0))
            pred = output.argmax(dim=1).item()
            if pred != label:
                return True, adv_image, modified_pixels
        
        # 计算梯度
        adv_image_batch = adv_image.unsqueeze(0).requires_grad_(True)
        output = model(adv_image_batch)
        
        # 对正确类别的置信度
        loss = output[0, label]
        
        model.zero_grad()
        loss.backward()
        grad = adv_image_batch.grad[0]
        
        # 更新动量
        momentum = beta * momentum + (1 - beta) * grad
        
        # 屏蔽已修改的像素
        momentum_masked = momentum.clone()
        momentum_masked[modified_mask] = 0
        
        # 找到累积梯度绝对值最大的像素
        momentum_abs = torch.abs(momentum_masked)
        max_idx = momentum_abs.argmax().item()
        
        # 转换为坐标（max_idx是int类型）
        c = max_idx // (H * W)
        h = (max_idx % (H * W)) // W
        w = max_idx % W
        
        # 如果梯度太小，停止
        if momentum_abs[c, h, w].item() < 1e-6:
            break
        
        # 计算扰动方向和大小
        grad_value = momentum[c, h, w].item()
        
        # 自适应步长：根据当前置信度调整
        with torch.no_grad():
            conf = torch.softmax(output[0], dim=0)[label].item()
            adaptive_alpha = alpha * (1 + conf)  # 置信度越高，步长越大
        
        # 应用扰动
        perturbation = -adaptive_alpha * torch.sign(torch.tensor(grad_value)).to(device)
        
        with torch.no_grad():
            original_value = adv_image[c, h, w].item()
            new_value = adv_image[c, h, w] + perturbation
            
            # 裁剪到[0, 1]
            new_value = torch.clamp(new_value, 0, 1)
            
            # 应用修改
            adv_image[c, h, w] = new_value
            
            # 标记为已修改
            modified_mask[c, h, w] = True
            # c, h, w 已经是int类型，不需要.item()
            modified_pixels.append((h, w, c))
    
    # 最后检查
    with torch.no_grad():
        output = model(adv_image.unsqueeze(0))
        pred = output.argmax(dim=1).item()
        if pred != label:
            return True, adv_image, modified_pixels
    
    return False, adv_image, modified_pixels


def test_pixel_gradient():
    """测试Pixel Gradient Attack"""
    import torchvision
    import torchvision.transforms as transforms
    from torch import nn
    
    print("="*80)
    print("🧪 测试 Pixel Gradient Attack")
    print("="*80)
    
    # 加载数据
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model.load_state_dict(torch.load('cifar10_resnet18.pth', map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    
    print(f"✅ 模型加载完成")
    print(f"🖥️  设备: {device}\n")
    
    # 测试10个样本
    success_count = 0
    total_l0 = 0
    
    print("开始测试10个样本...\n")
    
    for i in range(10):
        image, label = testset[i]
        
        # 确保初始预测正确
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            pred = output.argmax(dim=1).item()
            if pred != label:
                continue
        
        # 执行攻击
        success, adv_image, modified_pixels = pixel_gradient_attack(
            image=image,
            label=label,
            model=model,
            max_pixels=10,
            alpha=0.2,
            beta=0.9
        )
        
        if success:
            success_count += 1
            # 计算L0（去重）
            unique_pixels = set([(h, w) for h, w, c in modified_pixels])
            l0 = len(unique_pixels)
            total_l0 += l0
            print(f"✅ 样本{i}: 攻击成功, L0={l0}")
        else:
            print(f"❌ 样本{i}: 攻击失败")
    
    print(f"\n" + "="*80)
    print(f"📊 测试结果:")
    print(f"  ASR: {success_count}/10 = {success_count*10}%")
    if success_count > 0:
        print(f"  平均L0: {total_l0/success_count:.2f}")
    print("="*80)


if __name__ == "__main__":
    test_pixel_gradient()

