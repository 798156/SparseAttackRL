#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成5个方法的混淆矩阵横向排列图
顺序：JSMA, SparseFool, Greedy, PixelGrad, RandomSparse
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# CIFAR-10类别名称
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# 方法名称映射
METHOD_NAMES = {
    'jsma': 'JSMA',
    'sparsefool': 'SparseFool',
    'greedy': 'Greedy',
    'pixelgrad': 'PixelGrad',
    'randomsparse': 'RandomSparse'
}

# 方法顺序（按用户要求）
METHOD_ORDER = ['jsma', 'sparsefool', 'greedy', 'pixelgrad', 'randomsparse']

def load_confusion_matrices(json_path):
    """从JSON文件加载混淆矩阵数据"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 转换为numpy数组
    confusion_matrices = {}
    for method in METHOD_ORDER:
        if method in data:
            confusion_matrices[method] = np.array(data[method], dtype=int)
        else:
            print(f"⚠️  警告: 未找到方法 {method} 的数据")
            # 创建一个空的混淆矩阵
            confusion_matrices[method] = np.zeros((10, 10), dtype=int)
    
    return confusion_matrices

def normalize_confusion_matrix(confusion):
    """归一化混淆矩阵（按行转换为百分比）"""
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除零
    confusion_norm = confusion / row_sums * 100
    return confusion_norm

def create_5methods_confusion_matrix(confusion_matrices, output_dir):
    """创建5个方法的混淆矩阵横向排列图"""
    print("\n" + "="*60)
    print("📊 生成5个方法的混淆矩阵横向排列图...")
    print("="*60)
    
    # 创建图形，5个子图横向排列
    # 使用较大的图形尺寸以适应5个混淆矩阵
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    # 确保axes是数组
    if len(METHOD_ORDER) == 1:
        axes = [axes]
    
    # 为所有子图设置共享的颜色条范围
    vmin, vmax = 0, 100
    
    # 绘制每个方法的混淆矩阵
    for idx, method in enumerate(METHOD_ORDER):
        ax = axes[idx]
        confusion = confusion_matrices[method]
        
        # 归一化混淆矩阵
        confusion_norm = normalize_confusion_matrix(confusion)
        
        # 绘制热图
        # 只在最后一个子图显示颜色条
        cbar = (idx == len(METHOD_ORDER) - 1)
        
        sns.heatmap(
            confusion_norm,
            annot=True,
            fmt='.1f',
            cmap='Blues',
            cbar=cbar,
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            square=True,
            cbar_kws={'label': 'Percentage (%)', 'shrink': 0.8} if cbar else None
        )
        
        # 设置标题
        method_display = METHOD_NAMES[method]
        ax.set_title(
            f'{method_display} Confusion Matrix\n(% of successful attacks per class)',
            fontweight='bold',
            fontsize=11,
            pad=10
        )
        
        # 设置轴标签
        ax.set_xlabel('Adversarial Predicted Class', fontweight='bold', fontsize=10)
        if idx == 0:
            ax.set_ylabel('True Class', fontweight='bold', fontsize=10)
        else:
            ax.set_ylabel('')  # 只保留第一个子图的Y轴标签
        
        # 旋转x轴标签
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    png_path = output_dir / 'confusion_matrices_5methods.png'
    pdf_path = output_dir / 'confusion_matrices_5methods.pdf'
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"✓ PNG保存到: {png_path}")
    print(f"✓ PDF保存到: {pdf_path}")
    
    return png_path, pdf_path

def main():
    """主函数"""
    # 输入和输出路径
    json_path = Path('results/class_analysis/confusion_matrices.json')
    output_dir = Path('results/class_analysis')
    
    # 检查JSON文件是否存在
    if not json_path.exists():
        print(f"❌ 错误: 未找到混淆矩阵数据文件: {json_path}")
        print("   请先运行 generate_confusion_matrices.py 生成数据")
        return 1
    
    # 加载混淆矩阵数据
    print(f"📂 加载混淆矩阵数据: {json_path}")
    confusion_matrices = load_confusion_matrices(json_path)
    
    # 打印每个方法的成功攻击总数
    print("\n各方法成功攻击统计:")
    for method in METHOD_ORDER:
        total = confusion_matrices[method].sum()
        print(f"  {METHOD_NAMES[method]}: {total} 个成功攻击")
    
    # 生成5个方法的混淆矩阵图
    png_path, pdf_path = create_5methods_confusion_matrix(confusion_matrices, output_dir)
    
    print("\n" + "🎉"*30)
    print("5个方法的混淆矩阵横向排列图生成完成！")
    print("🎉"*30)
    print(f"\n📁 输出文件:")
    print(f"  1. {png_path}")
    print(f"  2. {pdf_path}")
    
    return 0

if __name__ == '__main__':
    exit(main())


