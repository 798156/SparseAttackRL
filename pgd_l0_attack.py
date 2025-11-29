"""
PGD-L0 Attack
基于投影梯度下降的L0稀疏攻击
每次迭代选择梯度最大的k个像素进行修改
"""

import torch
import numpy as np

def pgd_l0_attack(image, label, model, max_pixels=10, step_size=0.1, num_steps=20):
    """
    PGD-L0稀疏攻击
    
    原理：
    1. 计算梯度
    2. 选择梯度最大的k个像素
    3. 在这些像素上做PGD更新
    4. 投影回L0约束
    
    参数:
        image: 输入图像 (C, H, W)
        label: 真实标签
        model: 目标模型
        max_pixels: 最大修改像素数
        step_size: PGD步长
        num_steps: PGD迭代次数
    
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
    
    # 首先检查初始预测
    with torch.no_grad():
        output = model(adv_image.unsqueeze(0))
        pred = output.argmax(dim=1).item()
        if pred != label:
            return False, adv_image, []
    
    # 初始化：找到最重要的像素
    adv_image_batch = adv_image.unsqueeze(0).requires_grad_(True)
    output = model(adv_image_batch)
    loss = output[0, label]
    
    model.zero_grad()
    loss.backward()
    grad = adv_image_batch.grad[0]
    
    # 选择梯度绝对值最大的max_pixels个像素
    grad_abs = torch.abs(grad)
    grad_flat = grad_abs.view(-1)
    
    # 获取top-k索引
    C, H, W = adv_image.shape
    k = min(max_pixels * C, grad_flat.numel())  # max_pixels个像素 × 3通道
    _, topk_indices = torch.topk(grad_flat, k)
    
    # 创建掩码：只在这些位置上修改
    mask = torch.zeros_like(adv_image, dtype=torch.bool)
    for idx in topk_indices:
        c = idx // (H * W)
        h = (idx % (H * W)) // W
        w = idx % W
        mask[c, h, w] = True
    
    # 记录修改的像素位置（去重）
    modified_pixels_set = set()
    for idx in topk_indices:
        idx_item = idx.item()  # 先转换为int
        c = idx_item // (H * W)
        h = (idx_item % (H * W)) // W
        w = idx_item % W
        modified_pixels_set.add((h, w, c))
    
    # PGD迭代
    best_adv = adv_image.clone()
    best_conf = 1.0
    
    for step in range(num_steps):
        # 检查当前是否成功
        with torch.no_grad():
            output = model(adv_image.unsqueeze(0))
            pred = output.argmax(dim=1).item()
            
            if pred != label:
                return True, adv_image, list(modified_pixels_set)
            
            # 记录最佳（置信度最低的）
            conf = torch.softmax(output[0], dim=0)[label].item()
            if conf < best_conf:
                best_conf = conf
                best_adv = adv_image.clone()
        
        # 计算梯度
        adv_image_temp = adv_image.clone().requires_grad_(True)
        adv_image_batch = adv_image_temp.unsqueeze(0)
        
        output = model(adv_image_batch)
        loss = output[0, label]
        
        model.zero_grad()
        loss.backward()
        
        if adv_image_batch.grad is None:
            # 如果梯度计算失败，停止
            break
        
        grad = adv_image_batch.grad[0]
        
        # 只在mask位置上更新
        with torch.no_grad():
            # PGD更新
            perturbation = step_size * torch.sign(grad)
            adv_image = adv_image - perturbation * mask.float()
            
            # 投影到[0, 1]
            adv_image = torch.clamp(adv_image, 0, 1)
    
    # 最后检查一次best_adv
    with torch.no_grad():
        output = model(best_adv.unsqueeze(0))
        pred = output.argmax(dim=1).item()
        if pred != label:
            return True, best_adv, list(modified_pixels_set)
    
    return False, best_adv, list(modified_pixels_set)


def test_pgd_l0():
    """测试PGD-L0攻击"""
    import torchvision
    import torchvision.transforms as transforms
    from torch import nn
    
    print("="*80)
    print("🧪 测试 PGD-L0 Attack")
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
        success, adv_image, modified_pixels = pgd_l0_attack(
            image=image,
            label=label,
            model=model,
            max_pixels=10,
            step_size=0.1,
            num_steps=20
        )
        
        if success:
            success_count += 1
            l0 = len(set([(h, w) for h, w, c in modified_pixels]))
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
    test_pgd_l0()

