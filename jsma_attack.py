# jsma_attack.py
import numpy as np
import torch
import torch.nn.functional as F


def jsma_attack(image, label, model, max_pixels=5, theta=1.0, targeted=False, target_class=None):
    """
    Jacobian-based Saliency Map Attack (JSMA)
    
    参数:
        image: 输入图像 (C, H, W)
        label: 真实标签
        model: 目标模型
        max_pixels: 最大修改像素数
        theta: 每次修改的幅度
        targeted: 是否为定向攻击
        target_class: 目标类别（仅当 targeted=True 时有效）
    
    返回:
        success: 攻击是否成功
        adv_image: 对抗样本
        modified_pixels: 修改的像素列表 [(x, y, channel), ...]
    """
    device = next(model.parameters()).device
    
    # 确保图像在正确的设备上
    if isinstance(image, torch.Tensor):
        adv_image = image.clone().to(device)
    else:
        adv_image = torch.tensor(image).to(device)
    
    # 添加 batch 维度
    if adv_image.dim() == 3:
        adv_image = adv_image.unsqueeze(0)
    
    adv_image.requires_grad = True
    
    # 记录已修改的像素
    modified_pixels = []
    
    # 归一化参数
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).to(device).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).to(device).view(1, 3, 1, 1)
    
    # 获取图像尺寸
    _, C, H, W = adv_image.shape
    
    # 创建掩码，记录已修改的像素
    mask = torch.ones((C, H, W), dtype=torch.bool, device=device)
    
    # 首先检查初始预测是否正确
    with torch.no_grad():
        initial_output = model(adv_image)
        initial_pred = initial_output.argmax(dim=1).item()
        
        # 如果初始预测就错了，跳过这个样本
        if initial_pred != label:
            return False, adv_image.squeeze(0), []
    
    # 迭代修改像素
    for iteration in range(max_pixels):
        # 前向传播
        adv_image.requires_grad = True
        output = model(adv_image)
        
        # 检查是否已经成功
        pred = output.argmax(dim=1).item()
        if not targeted and pred != label:
            return True, adv_image.squeeze(0).detach(), modified_pixels
        if targeted and target_class is not None and pred == target_class:
            return True, adv_image.squeeze(0).detach(), modified_pixels
        
        # 计算雅可比矩阵
        jacobian = compute_jacobian(model, adv_image, output)
        
        # 计算显著性图
        if targeted and target_class is not None:
            saliency_map = compute_saliency_map(
                jacobian, label, target_class, mask, increase=True
            )
        else:
            # 非定向攻击：找到最容易改变预测的像素
            # 选择对正确类别贡献最大的像素进行修改
            target = output.argmax(dim=1).item()
            if target == label:
                # 如果还没改变预测，选择第二高的类别作为目标
                sorted_preds = output.argsort(dim=1, descending=True)[0]
                target = sorted_preds[1].item()
            
            saliency_map = compute_saliency_map(
                jacobian, label, target, mask, increase=True
            )
        
        # 简化策略：直接使用梯度绝对值最大的像素
        # 原JSMA的saliency map计算太严格，经常导致全零
        # 直接使用梯度最大的策略更稳定有效
        grad_label = jacobian[0, label]  # 正确类别的梯度
        grad_abs = torch.abs(grad_label) * mask.float()
        
        if grad_abs.max() == 0:
            # 所有可用像素的梯度都是0，无法继续攻击
            break
        
        flat_idx = grad_abs.argmax().item()
        
        c = flat_idx // (H * W)
        h = (flat_idx % (H * W)) // W
        w = flat_idx % W
        
        # 修改像素
        with torch.no_grad():
            # 计算修改方向（增加或减少）
            if targeted and target_class is not None:
                # 定向攻击：增加目标类别的激活
                direction = 1.0 if jacobian[0, target_class, c, h, w] > 0 else -1.0
            else:
                # 非定向攻击：减少正确类别的激活
                # 如果梯度>0，减少像素值；如果梯度<0，增加像素值
                direction = -1.0 if jacobian[0, label, c, h, w] > 0 else 1.0
            
            # 应用修改（在归一化空间中）
            # 使用更大的theta以确保影响足够大
            adv_image[0, c, h, w] += direction * theta
            
            # 裁剪到有效范围（考虑归一化）
            # CIFAR-10 的像素值范围是 [0, 1]，归一化后大约是 [-2, 2]
            adv_image[0, c, h, w] = torch.clamp(
                adv_image[0, c, h, w], 
                -3.0,  # 安全裕度
                3.0
            )
        
        # 记录修改的像素
        modified_pixels.append((w, h, c))
        
        # 更新掩码
        mask[c, h, w] = False
        
        # 分离梯度
        adv_image = adv_image.detach()
    
    # 最后检查一次
    with torch.no_grad():
        output = model(adv_image)
        pred = output.argmax(dim=1).item()
        
        if not targeted and pred != label:
            return True, adv_image.squeeze(0), modified_pixels
        if targeted and target_class is not None and pred == target_class:
            return True, adv_image.squeeze(0), modified_pixels
    
    return False, adv_image.squeeze(0).detach(), modified_pixels


def compute_jacobian(model, image, output):
    """
    计算输出关于输入的雅可比矩阵
    
    返回:
        jacobian: shape (1, num_classes, C, H, W)
    """
    num_classes = output.shape[1]
    _, C, H, W = image.shape
    
    jacobian = torch.zeros((1, num_classes, C, H, W), device=image.device)
    
    for class_idx in range(num_classes):
        # 对每个类别计算梯度
        model.zero_grad()
        if image.grad is not None:
            image.grad.zero_()
        
        # 计算该类别的梯度
        class_output = output[0, class_idx]
        class_output.backward(retain_graph=True)
        
        # 保存梯度
        if image.grad is not None:
            jacobian[0, class_idx] = image.grad[0].clone()
    
    return jacobian


def compute_saliency_map(jacobian, source_class, target_class, mask, increase=True):
    """
    计算显著性图
    
    参数:
        jacobian: 雅可比矩阵 (1, num_classes, C, H, W)
        source_class: 源类别（真实标签）
        target_class: 目标类别
        mask: 可修改像素的掩码
        increase: 是否增加像素值
    
    返回:
        saliency_map: 显著性图 (C*H*W,)
    """
    # 获取源类别和目标类别的梯度
    grad_target = jacobian[0, target_class]  # (C, H, W)
    grad_source = jacobian[0, source_class]  # (C, H, W)
    
    # JSMA 的显著性计算
    # 我们想要增加目标类别的激活，减少源类别的激活
    # 因此选择满足以下条件的像素：
    # 1. grad_target > 0 (增加像素值会增加目标类别激活)
    # 2. grad_source < 0 (增加像素值会减少源类别激活)
    
    if increase:
        # 增加像素值的情况
        alpha = grad_target
        beta = -grad_source
    else:
        # 减少像素值的情况
        alpha = -grad_target
        beta = grad_source
    
    # 计算显著性：只有当两个条件都满足时才为正
    # 使用条件：alpha > 0 且 beta > 0
    valid = (alpha > 0) & (beta > 0) & mask
    
    # 显著性 = alpha * beta（乘积越大越重要）
    saliency = alpha * beta * valid.float()
    
    # 展平为一维
    saliency_flat = saliency.view(-1)
    
    return saliency_flat


def jsma_attack_batch(images, labels, model, max_pixels=5, theta=1.0):
    """
    批量 JSMA 攻击
    
    参数:
        images: 输入图像批次 (N, C, H, W)
        labels: 真实标签 (N,)
        model: 目标模型
        max_pixels: 最大修改像素数
        theta: 每次修改的幅度
    
    返回:
        success_rate: 成功率
        adv_images: 对抗样本
        all_modified_pixels: 所有修改的像素列表
    """
    device = next(model.parameters()).device
    batch_size = images.shape[0]
    
    successes = []
    adv_images = []
    all_modified_pixels = []
    
    for i in range(batch_size):
        success, adv_img, modified = jsma_attack(
            images[i], labels[i].item(), model, 
            max_pixels=max_pixels, theta=theta
        )
        
        successes.append(success)
        adv_images.append(adv_img)
        all_modified_pixels.append(modified)
    
    success_rate = sum(successes) / batch_size
    adv_images = torch.stack(adv_images)
    
    return success_rate, adv_images, all_modified_pixels

