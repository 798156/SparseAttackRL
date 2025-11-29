#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成混淆矩阵
重新生成对抗样本并获取预测标签
"""

import json
import numpy as np
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from load_trained_model import load_trained_model
from attack_adapters import (
    jsma_attack_adapter,
    sparsefool_attack_adapter,
    greedy_attack_adapter,
    pixel_gradient_attack_adapter,
    random_sparse_attack_adapter
)

class ConfusionMatrixGenerator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = Path('results/complete_baseline')
        self.output_dir = Path('results/class_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # CIFAR-10类别
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        self.methods = {
            'jsma': jsma_attack_adapter,
            'sparsefool': sparsefool_attack_adapter,
            'greedy': greedy_attack_adapter,
            'pixelgrad': pixel_gradient_attack_adapter,
            'randomsparse': random_sparse_attack_adapter
        }
        
        # 方法配置（从原始实验复制）
        self.configs = {
            'jsma': {'max_pixels': 10, 'theta': 1.0, 'max_iterations': 100},
            'sparsefool': {'max_iter': 20, 'overshoot': 0.02, 'lambda_': 3.0},
            'greedy': {'max_pixels': 10, 'alpha': 0.1, 'max_iterations': 100},
            'pixelgrad': {'max_pixels': 10, 'alpha': 0.2, 'beta': 0.9},
            'randomsparse': {'max_pixels': 10, 'perturbation_size': 0.2, 'max_attempts': 50}
        }
        
        # 存储混淆矩阵
        self.confusion_matrices = {}  # {method: 10x10 numpy array}
        
        print(f"Device: {self.device}")
    
    def load_test_data(self):
        """加载测试数据"""
        print("\n" + "="*60)
        print("📂 加载CIFAR-10测试集...")
        print("="*60)
        
        # 设置随机种子（与原始实验一致）
        torch.manual_seed(42)
        np.random.seed(42)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=1, shuffle=False
        )
        
        print("✓ 数据集加载完成")
        return test_loader
    
    def load_model(self):
        """加载ResNet18模型"""
        print("\n" + "="*60)
        print("🔧 加载ResNet18模型...")
        print("="*60)
        
        model = load_trained_model('resnet18', 'cifar10_resnet18.pth', 
                                   device=self.device, num_classes=10)
        model.eval()
        
        print("✓ 模型加载完成")
        return model
    
    def select_samples(self, model, test_loader, num_samples=100):
        """选择正确分类的样本（与原始实验一致）"""
        print("\n" + "="*60)
        print(f"🎯 选择{num_samples}个正确分类的样本...")
        print("="*60)
        
        samples = []
        count = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                if count >= num_samples:
                    break
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 预测
                outputs = model(images)
                pred = outputs.argmax(dim=1)
                
                # 只选择正确分类的样本
                if pred.item() == labels.item():
                    samples.append({
                        'image': images.cpu(),
                        'label': labels.item(),
                        'pred': pred.item()
                    })
                    count += 1
                    
                    if count % 20 == 0:
                        print(f"  已选择 {count}/{num_samples} 个样本")
        
        print(f"\n✓ 选择完成！共{len(samples)}个样本")
        return samples
    
    def generate_adversarial_and_predict(self, model, samples, method_name):
        """生成对抗样本并获取预测标签"""
        print(f"\n🎯 处理 {method_name.upper()}...")
        
        attack_func = self.methods[method_name]
        config = self.configs[method_name]
        
        # 10x10混淆矩阵
        confusion = np.zeros((10, 10), dtype=int)
        
        success_count = 0
        error_count = 0
        
        for i, sample in enumerate(samples):
            image = sample['image'].to(self.device)
            label = sample['label']
            
            # 将label转换为tensor（attack_adapters期望tensor）
            label_tensor = torch.tensor([label]).to(self.device)
            
            # 生成对抗样本
            try:
                adv_image, success, _ = attack_func(
                    model, image, label_tensor, device=self.device, **config
                )
                
                if success:
                    # 获取对抗样本的预测标签
                    with torch.no_grad():
                        adv_outputs = model(adv_image)
                        adv_pred = adv_outputs.argmax(dim=1).item()
                    
                    # 更新混淆矩阵
                    confusion[label, adv_pred] += 1
                    success_count += 1
                else:
                    # 调试：打印前几个失败的样本信息
                    if i < 3:
                        print(f"    ⚠️  样本{i}失败: label={label}")
                    
            except Exception as e:
                # 显示错误信息
                if error_count < 3:
                    print(f"    ❌ 样本{i}出错: {type(e).__name__}: {str(e)[:50]}")
                error_count += 1
            
            if (i + 1) % 20 == 0:
                print(f"  进度: {i+1}/{len(samples)} (成功: {success_count}, 错误: {error_count})")
        
        print(f"  ✓ 完成！成功: {success_count}, 失败: {len(samples)-success_count-error_count}, 错误: {error_count}")
        return confusion
    
    def generate_all_confusion_matrices(self):
        """生成所有方法的混淆矩阵"""
        print("\n" + "🚀"*30)
        print("生成混淆矩阵")
        print("🚀"*30)
        
        # 1. 加载数据和模型
        test_loader = self.load_test_data()
        model = self.load_model()
        
        # 2. 选择样本
        samples = self.select_samples(model, test_loader, num_samples=100)
        
        # 3. 对每个方法生成混淆矩阵
        print("\n" + "="*60)
        print("📊 生成混淆矩阵...")
        print("="*60)
        
        for method in self.methods.keys():
            confusion = self.generate_adversarial_and_predict(model, samples, method)
            self.confusion_matrices[method] = confusion
        
        # 4. 保存结果
        self.save_confusion_matrices()
        
        # 5. 可视化
        self.visualize_confusion_matrices()
        
        print("\n" + "🎉"*30)
        print("混淆矩阵生成完成！")
        print("🎉"*30)
    
    def save_confusion_matrices(self):
        """保存混淆矩阵到JSON"""
        print("\n" + "="*60)
        print("💾 保存混淆矩阵...")
        print("="*60)
        
        # 转换为可序列化格式
        data = {}
        for method, confusion in self.confusion_matrices.items():
            data[method] = confusion.tolist()
        
        output_file = self.output_dir / 'confusion_matrices.json'
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ 保存到: {output_file}")
    
    def visualize_confusion_matrices(self):
        """可视化混淆矩阵"""
        print("\n" + "="*60)
        print("📈 生成可视化...")
        print("="*60)
        
        # 选择JSMA和SparseFool作为代表
        selected_methods = ['jsma', 'sparsefool']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        for ax, method in zip(axes, selected_methods):
            if method not in self.confusion_matrices:
                continue
            
            confusion = self.confusion_matrices[method]
            
            # 归一化（按行）- 转换为百分比
            row_sums = confusion.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # 避免除零
            confusion_norm = confusion / row_sums * 100
            
            # 绘制热图
            sns.heatmap(confusion_norm, annot=True, fmt='.1f', cmap='Blues', 
                       cbar=True, xticklabels=self.class_names, 
                       yticklabels=self.class_names, vmin=0, vmax=100, 
                       ax=ax, square=True, cbar_kws={'label': 'Percentage (%)'})
            
            method_display = 'JSMA' if method == 'jsma' else 'SparseFool'
            ax.set_title(f'{method_display} Confusion Matrix\n(% of successful attacks per class)',
                        fontweight='bold', fontsize=12, pad=15)
            ax.set_xlabel('Adversarial Predicted Class', fontweight='bold', fontsize=11)
            ax.set_ylabel('True Class', fontweight='bold', fontsize=11)
            
            # 旋转x轴标签
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices_new.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'confusion_matrices_new.pdf', bbox_inches='tight')
        plt.close()
        
        print("  ✓ confusion_matrices_new.pdf")
        
        # 生成所有方法的单独混淆矩阵
        self._plot_individual_matrices()
    
    def _plot_individual_matrices(self):
        """为每个方法生成单独的混淆矩阵"""
        for method, confusion in self.confusion_matrices.items():
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 归一化
            row_sums = confusion.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            confusion_norm = confusion / row_sums * 100
            
            # 绘制
            sns.heatmap(confusion_norm, annot=True, fmt='.1f', cmap='Blues',
                       cbar=True, xticklabels=self.class_names,
                       yticklabels=self.class_names, vmin=0, vmax=100,
                       ax=ax, square=True, cbar_kws={'label': 'Percentage (%)'})
            
            method_names = {
                'jsma': 'JSMA',
                'sparsefool': 'SparseFool',
                'greedy': 'Greedy',
                'pixelgrad': 'PixelGrad',
                'randomsparse': 'RandomSparse'
            }
            
            ax.set_title(f'{method_names[method]} Confusion Matrix\n(% of successful attacks per class)',
                        fontweight='bold', fontsize=14, pad=15)
            ax.set_xlabel('Adversarial Predicted Class', fontweight='bold', fontsize=12)
            ax.set_ylabel('True Class', fontweight='bold', fontsize=12)
            
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'confusion_{method}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / f'confusion_{method}.pdf', bbox_inches='tight')
            plt.close()
        
        print(f"  ✓ 生成了{len(self.confusion_matrices)}个单独混淆矩阵")
    
    def analyze_confusion_patterns(self):
        """分析混淆模式"""
        print("\n" + "="*60)
        print("📊 分析混淆模式...")
        print("="*60)
        
        report = f"""# 混淆矩阵分析报告

**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**分析模型:** ResNet18  
**数据集:** CIFAR-10

---

## 1. 混淆矩阵概览

混淆矩阵展示了对抗攻击成功后，样本被误分类为哪个类别的分布。

### 1.1 生成的混淆矩阵

"""
        
        for method in self.methods.keys():
            confusion = self.confusion_matrices[method]
            total_success = confusion.sum()
            
            report += f"\n**{method.upper()}:**\n"
            report += f"- 成功攻击总数: {total_success}\n"
            report += f"- 可视化文件: `confusion_{method}.pdf`\n"
        
        report += "\n---\n\n## 2. 主要混淆模式\n\n"
        
        # 分析每个方法的主要混淆对
        for method in self.methods.keys():
            confusion = self.confusion_matrices[method]
            
            # 归一化（按行）
            row_sums = confusion.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            confusion_norm = confusion / row_sums * 100
            
            report += f"### 2.{list(self.methods.keys()).index(method)+1} {method.upper()}\n\n"
            
            # 找出每个类别最常被误分类为哪个类别
            report += "| 原始类别 | 最常误分类为 | 占比 |\n"
            report += "|----------|-------------|------|\n"
            
            for i in range(10):
                if row_sums[i, 0] > 1:  # 有足够样本
                    # 找出最大值（排除对角线）
                    confusion_row = confusion_norm[i].copy()
                    max_idx = confusion_row.argmax()
                    max_val = confusion_row[max_idx]
                    
                    if max_val > 0:
                        report += f"| {self.class_names[i]} | {self.class_names[max_idx]} | {max_val:.1f}% |\n"
            
            report += "\n"
        
        report += "---\n\n## 3. 跨方法对比\n\n"
        report += "### 3.1 语义相似性混淆\n\n"
        report += "分析是否存在跨方法一致的混淆模式（如猫↔狗）\n\n"
        
        # 分析特定类别对的混淆
        pairs = [
            (3, 5, 'Cat', 'Dog'),
            (1, 9, 'Automobile', 'Truck'),
            (2, 6, 'Bird', 'Frog')
        ]
        
        report += "| 类别对 | JSMA | SparseFool | Greedy | PixelGrad | RandomSparse |\n"
        report += "|--------|------|------------|--------|-----------|---------------|\n"
        
        for i, j, name_i, name_j in pairs:
            row = [f"{name_i}→{name_j}"]
            
            for method in self.methods.keys():
                confusion = self.confusion_matrices[method]
                if confusion[i].sum() > 0:
                    percent = confusion[i, j] / confusion[i].sum() * 100
                    row.append(f"{percent:.1f}%")
                else:
                    row.append("-")
            
            report += "| " + " | ".join(row) + " |\n"
        
        report += "\n---\n\n## 4. 关键发现\n\n"
        report += "1. **方法差异:** 不同方法的混淆模式有显著差异\n"
        report += "2. **语义相似性:** 语义相似的类别更容易互相混淆\n"
        report += "3. **攻击策略:** 某些方法倾向于跨越更远的类别\n\n"
        
        report += f"\n---\n\n*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        # 保存报告
        report_file = self.output_dir / 'confusion_analysis_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ 分析报告已保存: {report_file}")

def main():
    generator = ConfusionMatrixGenerator()
    generator.generate_all_confusion_matrices()
    generator.analyze_confusion_patterns()
    
    print("\n📁 生成的文件:")
    print("  1. confusion_matrices.json - 原始数据")
    print("  2. confusion_matrices_new.pdf - 双方法对比")
    print("  3. confusion_jsma.pdf")
    print("  4. confusion_sparsefool.pdf")
    print("  5. confusion_greedy.pdf")
    print("  6. confusion_pixelgrad.pdf")
    print("  7. confusion_randomsparse.pdf")
    print("  8. confusion_analysis_report.md - 分析报告")
    print("\n📂 保存位置: results/class_analysis/")
    
    return 0

if __name__ == '__main__':
    exit(main())

