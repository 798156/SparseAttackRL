"""
Random Sparse Attack
完全随机选择像素进行扰动
作为baseline，证明智能方法的优越性
"""

import torch
import numpy as np

def random_sparse_attack(image, label, model, max_pixels=10, perturbation_size=0.3, max_attempts=100):
    """
    随机稀疏攻击
    
    原理：
    1. 随机选择k个像素
    2. 随机方向扰动（±perturbation_size）
    3. 重复多次尝试，找到成功的组合
    
    这是最简单的稀疏攻击baseline
    用于证明智能方法（JSMA, Greedy等）的必要性
    
    参数:
        image: 输入图像 (C, H, W)
        label: 真实标签
        model: 目标模型
        max_pixels: 最大修改像素数
        perturbation_size: 扰动大小
        max_attempts: 最大尝试次数
    
    返回:
        success: 是否成功
        adv_image: 对抗样本
        modified_pixels: 修改的像素列表
    """
    device = next(model.parameters()).device
    original_image = image.clone().to(device)
    
    # 首先检查初始预测
    with torch.no_grad():
        output = model(original_image.unsqueeze(0))
        pred = output.argmax(dim=1).item()
        if pred != label:
            return False, original_image, []
    
    C, H, W = image.shape
    total_pixels = H * W
    
    # 多次随机尝试
    for attempt in range(max_attempts):
        # 随机选择像素
        num_pixels = np.random.randint(1, max_pixels + 1)  # 随机选择1到max_pixels个
        selected_pixels = np.random.choice(total_pixels, size=num_pixels, replace=False)
        
        # 创建对抗样本
        adv_image = original_image.clone()
        modified_pixels = []
        
        for pixel_idx in selected_pixels:
            h = pixel_idx // W
            w = pixel_idx % W
            
            # 随机选择通道
            channels = np.random.choice([0, 1, 2], size=np.random.randint(1, 4), replace=False)
            
            for c in channels:
                # 随机方向扰动
                perturbation = np.random.choice([-1, 1]) * perturbation_size
                
                # 应用扰动
                new_value = adv_image[c, h, w] + perturbation
                adv_image[c, h, w] = torch.clamp(new_value, 0, 1)
                
                modified_pixels.append((h, w, c))
        
        # 测试是否成功
        with torch.no_grad():
            output = model(adv_image.unsqueeze(0))
            pred = output.argmax(dim=1).item()
            
            if pred != label:
                # 成功！返回
                # 去重像素位置
                unique_pixels = list(set([(h, w, c) for h, w, c in modified_pixels]))
                return True, adv_image, unique_pixels
    
    # 如果所有尝试都失败，返回最后一次尝试的结果
    return False, adv_image, []


def random_sparse_attack_smart(image, label, model, max_pixels=10, perturbation_size=0.2, max_attempts=50):
    """
    改进的随机稀疏攻击
    使用梯度信息来确定扰动方向（但像素选择仍然是随机的）
    
    这个版本介于完全随机和智能方法之间
    """
    device = next(model.parameters()).device
    original_image = image.clone().to(device)
    
    # 首先检查初始预测
    with torch.no_grad():
        output = model(original_image.unsqueeze(0))
        pred = output.argmax(dim=1).item()
        if pred != label:
            return False, original_image, []
    
    C, H, W = image.shape
    total_pixels = H * W
    
    # 计算一次梯度，用于确定扰动方向
    adv_temp = original_image.clone().requires_grad_(True)
    output = model(adv_temp.unsqueeze(0))
    loss = output[0, label]
    
    model.zero_grad()
    loss.backward()
    grad = adv_temp.grad
    
    # 多次随机尝试
    for attempt in range(max_attempts):
        # 随机选择像素位置
        num_pixels = np.random.randint(1, max_pixels + 1)
        selected_positions = np.random.choice(total_pixels, size=num_pixels, replace=False)
        
        # 创建对抗样本
        adv_image = original_image.clone()
        modified_pixels = []
        
        for pos in selected_positions:
            h = pos // W
            w = pos % W
            
            # 对所有通道应用扰动（使用梯度方向）
            for c in range(C):
                if grad is not None:
                    # 使用梯度方向
                    perturbation = -perturbation_size * torch.sign(grad[c, h, w])
                else:
                    # 如果没有梯度，随机方向
                    perturbation = np.random.choice([-1, 1]) * perturbation_size
                
                new_value = adv_image[c, h, w] + perturbation
                adv_image[c, h, w] = torch.clamp(new_value, 0, 1)
                
                modified_pixels.append((h, w, c))
        
        # 测试是否成功
        with torch.no_grad():
            output = model(adv_image.unsqueeze(0))
            pred = output.argmax(dim=1).item()
            
            if pred != label:
                unique_pixels = list(set([(h, w, c) for h, w, c in modified_pixels]))
                return True, adv_image, unique_pixels
    
    return False, adv_image, []


def test_random_sparse():
    """测试Random Sparse Attack"""
    import torchvision
    import torchvision.transforms as transforms
    from torch import nn
    
    print("="*80)
    print("🧪 测试 Random Sparse Attack")
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
    
    # 测试两个版本
    print("=" * 80)
    print("版本1：完全随机")
    print("=" * 80)
    
    success_count = 0
    total_l0 = 0
    
    for i in range(10):
        image, label = testset[i]
        
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            pred = output.argmax(dim=1).item()
            if pred != label:
                continue
        
        success, adv_image, modified_pixels = random_sparse_attack(
            image=image, label=label, model=model,
            max_pixels=10, perturbation_size=0.3, max_attempts=100
        )
        
        if success:
            success_count += 1
            unique_pixels = set([(h, w) for h, w, c in modified_pixels])
            l0 = len(unique_pixels)
            total_l0 += l0
            print(f"✅ 样本{i}: 成功, L0={l0}")
        else:
            print(f"❌ 样本{i}: 失败")
    
    print(f"\n完全随机版本 ASR: {success_count}/10 = {success_count*10}%")
    if success_count > 0:
        print(f"平均L0: {total_l0/success_count:.2f}")
    
    # 测试智能版本
    print("\n" + "=" * 80)
    print("版本2：随机位置 + 梯度方向")
    print("=" * 80)
    
    success_count2 = 0
    total_l0_2 = 0
    
    for i in range(10):
        image, label = testset[i]
        
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            pred = output.argmax(dim=1).item()
            if pred != label:
                continue
        
        success, adv_image, modified_pixels = random_sparse_attack_smart(
            image=image, label=label, model=model,
            max_pixels=10, perturbation_size=0.2, max_attempts=50
        )
        
        if success:
            success_count2 += 1
            unique_pixels = set([(h, w) for h, w, c in modified_pixels])
            l0 = len(unique_pixels)
            total_l0_2 += l0
            print(f"✅ 样本{i}: 成功, L0={l0}")
        else:
            print(f"❌ 样本{i}: 失败")
    
    print(f"\n智能版本 ASR: {success_count2}/10 = {success_count2*10}%")
    if success_count2 > 0:
        print(f"平均L0: {total_l0_2/success_count2:.2f}")
    
    print("\n" + "="*80)
    print("💡 对比:")
    print(f"  完全随机: {success_count*10}% ASR")
    print(f"  随机+梯度: {success_count2*10}% ASR")
    print("="*80)


if __name__ == "__main__":
    test_random_sparse()
















