# visualization.py
import matplotlib.pyplot as plt
import numpy as np

def tensor_to_numpy(img_tensor):
    """
    将 PyTorch Tensor 转为可显示的 numpy 图像
    """
    # ✅ 关键修复：添加 .detach()
    img = img_tensor.permute(1, 2, 0).cpu().detach().numpy()

    # 反归一化
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    img = std * img + mean

    # 限制范围
    img = np.clip(img, 0, 1)
    return img


def plot_attack_comparison(original, adversarial, modified_coord=None):
    """
    并排显示原始图和对抗图
    """
    plt.figure(figsize=(10, 5))

    # 左图：原始图像
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(tensor_to_numpy(original))
    plt.axis('off')

    # 右图：对抗图像
    plt.subplot(1, 2, 2)
    plt.title("Adversarial Image")
    plt.imshow(tensor_to_numpy(adversarial))

    # 标记修改的像素
    if modified_coord:
        x, y = modified_coord
        plt.scatter(x, y, color='red', s=100, marker='x', linewidths=3)

    plt.axis('off')
    plt.tight_layout()
    plt.show()
