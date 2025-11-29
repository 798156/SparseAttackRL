"""
Foolbox官方攻击方法的封装
使用官方实现确保baseline的准确性和可信度
"""

import torch
import numpy as np
import foolbox as fb
from foolbox.attacks import FGSM, PGD, L2DeepFoolAttack, BoundaryAttack, L2CarliniWagnerAttack


def create_foolbox_model(model, bounds=(0, 1), preprocessing=None):
    """
    创建Foolbox模型包装器
    
    Args:
        model: PyTorch模型
        bounds: 输入范围
        preprocessing: 预处理字典
    """
    if preprocessing is None:
        # CIFAR-10的标准化参数
        preprocessing = {
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'axis': -3  # channels first
        }
    
    fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)
    return fmodel


def foolbox_jsma_attack(image, label, model, max_pixels=5, device='cuda'):
    """
    使用Foolbox的官方JSMA实现
    注意：Foolbox 3.x 没有直接的JSMA实现，我们使用DeepFool作为替代
    
    Args:
        image: 输入图像 [C, H, W]
        label: 真实标签
        model: PyTorch模型
        max_pixels: 最大修改像素数（用于后处理）
        device: 设备
    
    Returns:
        success: 是否成功
        adv_image: 对抗样本
        modified_pixels: 修改的像素位置列表
    """
    # 检查初始预测
    with torch.no_grad():
        initial_output = model(image.unsqueeze(0))
        initial_pred = initial_output.argmax(dim=1).item()
        
        if initial_pred != label:
            return False, image, []
    
    # 创建Foolbox模型
    fmodel = create_foolbox_model(model)
    
    # DeepFool是一个好的替代方案（迭代式，修改较少像素）
    attack = L2DeepFoolAttack()
    
    # 转换为numpy
    image_np = image.cpu().numpy()
    label_np = np.array([label])
    
    # 执行攻击
    try:
        _, adv_np, success = attack(fmodel, image_np[np.newaxis, ...], label_np, epsilons=None)
        
        if not success[0]:
            return False, image, []
        
        # 转换回torch
        adv_image = torch.from_numpy(adv_np[0]).to(device)
        
        # 验证攻击成功
        with torch.no_grad():
            output = model(adv_image.unsqueeze(0))
            pred = output.argmax(dim=1).item()
            
            if pred == label:
                return False, image, []
        
        # 找到修改的像素
        diff = torch.abs(adv_image - image)
        modified_mask = diff.sum(dim=0) > 1e-6
        modified_pixels = modified_mask.nonzero(as_tuple=False).cpu().tolist()
        
        return True, adv_image, modified_pixels
        
    except Exception as e:
        print(f"Foolbox DeepFool攻击失败: {e}")
        return False, image, []


def foolbox_boundary_attack(image, label, model, max_iterations=100, device='cuda'):
    """
    使用Foolbox的Boundary Attack（决策边界攻击，适合稀疏攻击）
    
    Args:
        image: 输入图像 [C, H, W]
        label: 真实标签
        model: PyTorch模型
        max_iterations: 最大迭代次数
        device: 设备
    
    Returns:
        success: 是否成功
        adv_image: 对抗样本
        modified_pixels: 修改的像素位置列表
    """
    # 检查初始预测
    with torch.no_grad():
        initial_output = model(image.unsqueeze(0))
        initial_pred = initial_output.argmax(dim=1).item()
        
        if initial_pred != label:
            return False, image, []
    
    # 创建Foolbox模型
    fmodel = create_foolbox_model(model)
    
    # Boundary Attack
    attack = BoundaryAttack(steps=max_iterations)
    
    # 转换为numpy
    image_np = image.cpu().numpy()
    label_np = np.array([label])
    
    # 执行攻击
    try:
        _, adv_np, success = attack(fmodel, image_np[np.newaxis, ...], label_np, epsilons=None)
        
        if not success[0]:
            return False, image, []
        
        # 转换回torch
        adv_image = torch.from_numpy(adv_np[0]).to(device)
        
        # 验证攻击成功
        with torch.no_grad():
            output = model(adv_image.unsqueeze(0))
            pred = output.argmax(dim=1).item()
            
            if pred == label:
                return False, image, []
        
        # 找到修改的像素
        diff = torch.abs(adv_image - image)
        modified_mask = diff.sum(dim=0) > 1e-6
        modified_pixels = modified_mask.nonzero(as_tuple=False).cpu().tolist()
        
        return True, adv_image, modified_pixels
        
    except Exception as e:
        print(f"Foolbox Boundary攻击失败: {e}")
        return False, image, []


def foolbox_cw_attack(image, label, model, confidence=0, max_iterations=100, device='cuda'):
    """
    使用Foolbox的C&W L2攻击
    
    Args:
        image: 输入图像 [C, H, W]
        label: 真实标签
        model: PyTorch模型
        confidence: 置信度参数
        max_iterations: 最大迭代次数
        device: 设备
    
    Returns:
        success: 是否成功
        adv_image: 对抗样本
        modified_pixels: 修改的像素位置列表
    """
    # 检查初始预测
    with torch.no_grad():
        initial_output = model(image.unsqueeze(0))
        initial_pred = initial_output.argmax(dim=1).item()
        
        if initial_pred != label:
            return False, image, []
    
    # 创建Foolbox模型
    fmodel = create_foolbox_model(model)
    
    # C&W Attack
    attack = L2CarliniWagnerAttack(
        binary_search_steps=9,
        steps=max_iterations,
        stepsize=0.01,
        confidence=confidence,
        initial_const=0.001,
        abort_early=True
    )
    
    # 转换为numpy
    image_np = image.cpu().numpy()
    label_np = np.array([label])
    
    # 执行攻击
    try:
        _, adv_np, success = attack(fmodel, image_np[np.newaxis, ...], label_np, epsilons=None)
        
        if not success[0]:
            return False, image, []
        
        # 转换回torch
        adv_image = torch.from_numpy(adv_np[0]).to(device)
        
        # 验证攻击成功
        with torch.no_grad():
            output = model(adv_image.unsqueeze(0))
            pred = output.argmax(dim=1).item()
            
            if pred == label:
                return False, image, []
        
        # 找到修改的像素
        diff = torch.abs(adv_image - image)
        modified_mask = diff.sum(dim=0) > 1e-6
        modified_pixels = modified_mask.nonzero(as_tuple=False).cpu().tolist()
        
        return True, adv_image, modified_pixels
        
    except Exception as e:
        print(f"Foolbox C&W攻击失败: {e}")
        return False, image, []


def foolbox_fgsm_attack(image, label, model, epsilon=0.1, device='cuda'):
    """
    使用Foolbox的FGSM攻击
    
    Args:
        image: 输入图像 [C, H, W]
        label: 真实标签
        model: PyTorch模型
        epsilon: 扰动大小
        device: 设备
    
    Returns:
        success: 是否成功
        adv_image: 对抗样本
        modified_pixels: 修改的像素位置列表
    """
    # 检查初始预测
    with torch.no_grad():
        initial_output = model(image.unsqueeze(0))
        initial_pred = initial_output.argmax(dim=1).item()
        
        if initial_pred != label:
            return False, image, []
    
    # 创建Foolbox模型
    fmodel = create_foolbox_model(model)
    
    # FGSM Attack
    attack = FGSM()
    
    # 转换为numpy
    image_np = image.cpu().numpy()
    label_np = np.array([label])
    
    # 执行攻击
    try:
        _, adv_np, success = attack(fmodel, image_np[np.newaxis, ...], label_np, epsilons=epsilon)
        
        if not success[0]:
            return False, image, []
        
        # 转换回torch
        adv_image = torch.from_numpy(adv_np[0]).to(device)
        
        # 验证攻击成功
        with torch.no_grad():
            output = model(adv_image.unsqueeze(0))
            pred = output.argmax(dim=1).item()
            
            if pred == label:
                return False, image, []
        
        # 找到修改的像素
        diff = torch.abs(adv_image - image)
        modified_mask = diff.sum(dim=0) > 1e-6
        modified_pixels = modified_mask.nonzero(as_tuple=False).cpu().tolist()
        
        return True, adv_image, modified_pixels
        
    except Exception as e:
        print(f"Foolbox FGSM攻击失败: {e}")
        return False, image, []


# 使用说明
if __name__ == "__main__":
    print("=" * 80)
    print("📚 Foolbox官方攻击方法封装")
    print("=" * 80)
    print("""
    本模块提供了以下Foolbox官方攻击方法的封装：
    
    1. DeepFool Attack (替代JSMA)
       - 迭代式攻击，寻找最小扰动
       - 适合稀疏攻击场景
       - 使用: foolbox_jsma_attack()
    
    2. Boundary Attack
       - 决策边界攻击
       - 不需要梯度信息
       - 使用: foolbox_boundary_attack()
    
    3. C&W L2 Attack
       - 经典的优化基攻击
       - 生成高质量对抗样本
       - 使用: foolbox_cw_attack()
    
    4. FGSM Attack
       - 快速梯度符号攻击
       - 最快的攻击方法
       - 使用: foolbox_fgsm_attack()
    
    推荐用于论文对比：
    - DeepFool: 作为JSMA的替代（Foolbox 3.x没有JSMA）
    - C&W: 经典强基准
    - Boundary: 黑盒攻击基准
    """)

