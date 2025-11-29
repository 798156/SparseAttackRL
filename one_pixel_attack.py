# one_pixel_attack.py
import numpy as np
from scipy.optimize import differential_evolution
import torch


def one_pixel_attack(image, label, model, max_iter=100, pixels=1, pop_size=10):
    """
    使用差分进化算法进行 One-Pixel Attack
    只允许修改 1 个像素，看能否欺骗模型
    
    参数:
        image: 输入图像 (C, H, W)
        label: 真实标签
        model: 目标模型
        max_iter: 最大迭代次数
        pixels: 修改像素数（默认1）
        pop_size: 种群大小（默认10）
    """
    device = next(model.parameters()).device
    image = image.unsqueeze(0).to(device)  # 添加 batch 维度

    # ✅ 关键修复：将 int 转换为 Tensor
    label_tensor = torch.tensor(label).to(device)

    def fitness_function(params):
        """
        优化目标函数
        params = [x, y, r, g, b]
        """
        x, y = int(params[0]), int(params[1])
        r, g, b = params[2], params[3], params[4]

        # 构造对抗图像（复制原图）
        adv_img = image.clone()

        # 归一化参数
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]

        # 修改指定像素（先反归一化 → 改 → 再归一化）
        for c in range(3):
            val = (r, g, b)[c]
            pixel_norm = (val / 255.0 - mean[c]) / std[c]
            adv_img[0, c, y, x] = pixel_norm

        # 推理
        with torch.no_grad():
            output = model(adv_img)
            pred = output.argmax(dim=1).item()

        # 成功返回负损失，失败返回正损失（优化器最小化）
        if pred != label_tensor.item():  # ✅ 使用 label_tensor
            return -1.0  # 成功
        else:
            # 否则返回交叉熵损失（越小越好）
            loss = -torch.log_softmax(output, dim=1)[0, label_tensor.item()].item()
            return loss

    # 差分进化搜索最优解
    # bounds: [(x_min, x_max), (y_min, y_max), (r_min, r_max), ...]
    bounds = [(0, 31), (0, 31)] + [(0, 255)] * 3  # CIFAR-10 是 32x32，像素值 0~255

    result = differential_evolution(
        fitness_function,
        bounds,
        maxiter=max_iter,
        popsize=pop_size,  # ✅ 使用可配置的种群大小
        tol=0.01,
        seed=None  # ✅ 改为None以获得更多随机性
    )

    # 判断是否成功
    success = fitness_function(result.x) < 0
    
    if success:
        # 构造最终的对抗样本
        x, y = int(result.x[0]), int(result.x[1])
        r, g, b = result.x[2], result.x[3], result.x[4]
        
        adv_img = image.clone()
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        
        for c in range(3):
            val = (r, g, b)[c]
            pixel_norm = (val / 255.0 - mean[c]) / std[c]
            adv_img[0, c, y, x] = pixel_norm
        
        modified_info = {'x': x, 'y': y, 'r': r, 'g': g, 'b': b}
        return success, adv_img.squeeze(0).cpu(), modified_info
    else:
        return False, image.squeeze(0).cpu(), {}
