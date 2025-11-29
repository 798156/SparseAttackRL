# evaluation_metrics.py
"""
扩展的评估指标计算
用于论文实验的全面评估
"""

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def compute_l0_norm(original, adversarial, threshold=1e-6):
    """
    计算L0范数：修改的像素数量
    
    Args:
        original: 原始图像 (C, H, W)
        adversarial: 对抗图像 (C, H, W)
        threshold: 判定像素被修改的阈值
    
    Returns:
        l0: 修改的像素数
    """
    # 使用阈值判断，而不是严格的不等于（避免浮点数精度问题）
    diff = torch.abs(original - adversarial)
    # 如果任何通道的差异超过阈值，该像素就算被修改
    l0 = (diff.sum(dim=0) > threshold).sum().item()
    return l0


def compute_l2_norm(original, adversarial):
    """
    计算L2范数：扰动的欧式距离
    
    Args:
        original: 原始图像 (C, H, W)
        adversarial: 对抗图像 (C, H, W)
    
    Returns:
        l2: L2范数
    """
    diff = (original - adversarial).flatten()
    l2 = torch.norm(diff, p=2).item()
    return l2


def compute_linf_norm(original, adversarial):
    """
    计算L∞范数：最大单像素扰动
    
    Args:
        original: 原始图像 (C, H, W)
        adversarial: 对抗图像 (C, H, W)
    
    Returns:
        linf: L∞范数
    """
    diff = torch.abs(original - adversarial)
    linf = diff.max().item()
    return linf


def compute_ssim(original, adversarial):
    """
    计算结构相似度 (SSIM)
    值越接近1表示图像越相似
    
    Args:
        original: 原始图像 (C, H, W) Tensor
        adversarial: 对抗图像 (C, H, W) Tensor
    
    Returns:
        ssim_value: SSIM值 (0-1)
    """
    # 转换为numpy并调整维度顺序 (H, W, C)
    orig_np = original.cpu().numpy().transpose(1, 2, 0)
    adv_np = adversarial.cpu().numpy().transpose(1, 2, 0)
    
    # 归一化到[0, 1]范围
    orig_np = (orig_np - orig_np.min()) / (orig_np.max() - orig_np.min() + 1e-8)
    adv_np = (adv_np - adv_np.min()) / (adv_np.max() - adv_np.min() + 1e-8)
    
    # 计算SSIM
    ssim_value = ssim(orig_np, adv_np, multichannel=True, channel_axis=2, data_range=1.0)
    
    return ssim_value


def compute_psnr(original, adversarial):
    """
    计算峰值信噪比 (PSNR)
    值越大表示图像质量越好
    
    Args:
        original: 原始图像 (C, H, W) Tensor
        adversarial: 对抗图像 (C, H, W) Tensor
    
    Returns:
        psnr_value: PSNR值 (dB)
    """
    # 转换为numpy
    orig_np = original.cpu().numpy().transpose(1, 2, 0)
    adv_np = adversarial.cpu().numpy().transpose(1, 2, 0)
    
    # 归一化到[0, 1]范围
    orig_np = (orig_np - orig_np.min()) / (orig_np.max() - orig_np.min() + 1e-8)
    adv_np = (adv_np - adv_np.min()) / (adv_np.max() - adv_np.min() + 1e-8)
    
    # 计算PSNR
    psnr_value = psnr(orig_np, adv_np, data_range=1.0)
    
    return psnr_value


def compute_all_metrics(original, adversarial):
    """
    计算所有评估指标
    
    Args:
        original: 原始图像 (C, H, W)
        adversarial: 对抗图像 (C, H, W)
    
    Returns:
        metrics: 包含所有指标的字典
    """
    metrics = {
        'l0_norm': compute_l0_norm(original, adversarial),
        'l2_norm': compute_l2_norm(original, adversarial),
        'linf_norm': compute_linf_norm(original, adversarial),
        'ssim': compute_ssim(original, adversarial),
        'psnr': compute_psnr(original, adversarial)
    }
    
    return metrics


def compute_query_efficiency(num_queries, success):
    """
    计算查询效率
    
    Args:
        num_queries: 查询次数
        success: 是否成功
    
    Returns:
        efficiency: 效率分数（成功时为1/queries，失败时为0）
    """
    if success:
        return 1.0 / num_queries
    else:
        return 0.0


class MetricsAggregator:
    """聚合多个样本的指标统计"""
    
    def __init__(self):
        self.metrics = {
            'l0_norm': [],
            'l2_norm': [],
            'linf_norm': [],
            'ssim': [],
            'psnr': [],
            'query_count': [],
            'attack_time': [],
            'success': []
        }
    
    def add(self, **kwargs):
        """添加一个样本的指标"""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def compute_statistics(self):
        """计算统计信息"""
        stats = {}
        
        for metric_name, values in self.metrics.items():
            if not values:
                continue
            
            if metric_name == 'success':
                # 布尔值：计算成功率
                stats[f'{metric_name}_rate'] = np.mean(values)
            else:
                # 数值：计算均值、标准差、中位数
                successful_values = [v for i, v in enumerate(values) 
                                   if self.metrics['success'][i]]
                
                if successful_values:
                    stats[f'{metric_name}_mean'] = np.mean(successful_values)
                    stats[f'{metric_name}_std'] = np.std(successful_values)
                    stats[f'{metric_name}_median'] = np.median(successful_values)
                    stats[f'{metric_name}_min'] = np.min(successful_values)
                    stats[f'{metric_name}_max'] = np.max(successful_values)
        
        return stats
    
    def save_to_csv(self, filename):
        """保存原始数据到CSV"""
        import pandas as pd
        
        # 确保所有列长度一致（用None填充缺失值）
        max_len = max(len(v) for v in self.metrics.values() if v)
        
        aligned_metrics = {}
        for key, values in self.metrics.items():
            if len(values) < max_len:
                # 用None填充到最大长度
                aligned_metrics[key] = values + [None] * (max_len - len(values))
            else:
                aligned_metrics[key] = values
        
        df = pd.DataFrame(aligned_metrics)
        df.to_csv(filename, index=False)
        print(f"✅ 指标数据已保存至: {filename}")


def statistical_significance_test(method1_results, method2_results, metric='asr'):
    """
    统计显著性检验（配对t检验）
    
    Args:
        method1_results: 方法1的结果列表（布尔或数值）
        method2_results: 方法2的结果列表（布尔或数值）
        metric: 评估指标名称
    
    Returns:
        result: 包含t统计量、p值、是否显著的字典
    """
    from scipy import stats
    
    # 转换为数值数组（如果是布尔类型）
    method1_array = np.array(method1_results, dtype=float)
    method2_array = np.array(method2_results, dtype=float)
    
    # 配对t检验
    t_stat, p_value = stats.ttest_rel(method1_array, method2_array)
    
    # 判断显著性（通常使用 α=0.05）
    is_significant = p_value < 0.05
    
    result = {
        'metric': metric,
        't_statistic': t_stat,
        'p_value': p_value,
        'is_significant': is_significant,
        'alpha': 0.05
    }
    
    # 打印结果
    print(f"\n📊 {metric} 的统计显著性检验:")
    print(f"   t统计量: {t_stat:.4f}")
    print(f"   p值: {p_value:.4f}")
    print(f"   显著性(α=0.05): {'✅ 显著' if is_significant else '❌ 不显著'}")
    
    return result


# 使用示例
if __name__ == "__main__":
    print("📊 评估指标模块")
    print("=" * 50)
    
    # 模拟示例
    original = torch.randn(3, 32, 32)
    adversarial = original.clone()
    adversarial[:, 10, 10] += 0.5  # 修改一个像素
    
    # 计算所有指标
    metrics = compute_all_metrics(original, adversarial)
    
    print("\n指标计算结果:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # 聚合器示例
    print("\n聚合器示例:")
    aggregator = MetricsAggregator()
    
    # 添加几个样本
    for i in range(5):
        aggregator.add(
            l0_norm=i+1,
            l2_norm=np.random.rand(),
            success=i % 2 == 0,
            attack_time=np.random.rand() * 10
        )
    
    # 计算统计
    stats = aggregator.compute_statistics()
    print("\n统计结果:")
    for name, value in stats.items():
        print(f"  {name}: {value:.4f}")



