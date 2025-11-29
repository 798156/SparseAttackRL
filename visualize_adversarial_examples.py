#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对抗样本可视化脚本
生成高质量的对抗样本对比图和像素修改热图
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json

# 导入攻击方法
from jsma_attack import jsma_attack
from sparsefool_attack import sparsefool_attack
from greedy_attack import greedy_attack
from pixel_gradient_attack import pixel_gradient_attack
from random_sparse_attack import random_sparse_attack_smart

# 导入模型和数据
from load_trained_model import load_trained_model
from dataset_loader import DatasetLoader

class AdversarialVisualizer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.output_dir = Path('results/adversarial_visualization')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # CIFAR-10类别名称
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        self.class_names_cn = [
            '飞机', '汽车', '鸟', '猫', '鹿',
            '狗', '青蛙', '马', '船', '卡车'
        ]
        
        # 加载模型
        print("Loading model...")
        self.model = load_trained_model('resnet18', 'cifar10_resnet18.pth', self.device)
        self.model.eval()
        
        # 加载数据
        print("Loading data...")
        dataset_loader = DatasetLoader('cifar10', './data')
        self.test_loader = dataset_loader.load_test_set()
        
        # CIFAR-10 denormalization
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        self.std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    
    def denormalize(self, tensor):
        """反归一化图像用于显示"""
        tensor = tensor.cpu().clone()
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        tensor = tensor * self.std + self.mean
        tensor = torch.clamp(tensor, 0, 1)
        return tensor
    
    def tensor_to_image(self, tensor):
        """将tensor转换为可显示的numpy数组"""
        tensor = self.denormalize(tensor)
        return tensor.permute(1, 2, 0).numpy()
    
    def select_samples_for_visualization(self, num_samples=5):
        """选择用于可视化的样本"""
        print(f"\nSelecting {num_samples} samples for visualization...")
        
        samples = []
        count = 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                if count >= num_samples:
                    break
                
                images = images.to(self.device)
                
                # 确保labels是tensor
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor([labels])
                labels = labels.to(self.device)
                
                # 确保images有batch维度
                if images.dim() == 3:
                    images = images.unsqueeze(0)
                
                # 确保正确分类
                outputs = self.model(images)
                pred = outputs.argmax(dim=1)
                
                if pred.item() == labels.item():
                    samples.append({
                        'image': images,
                        'label': labels.item(),
                        'pred': pred.item(),
                        'confidence': torch.softmax(outputs, dim=1)[0, pred].item()
                    })
                    count += 1
                    print(f"  Selected sample {count}: {self.class_names[labels.item()]} (conf: {samples[-1]['confidence']:.2f})")
        
        return samples
    
    def generate_adversarial_examples(self, sample):
        """为一个样本生成所有方法的对抗样本"""
        image = sample['image']
        label = torch.tensor([sample['label']]).to(self.device)
        
        results = {}
        
        print(f"\n  Generating adversarial examples...")
        
        # JSMA
        try:
            success, adv_jsma, modified_jsma = jsma_attack(
                image.squeeze(0), label.item(), self.model,
                max_pixels=10, theta=1.0
            )
            if success:
                results['JSMA'] = {
                    'image': adv_jsma.unsqueeze(0),
                    'modified_pixels': modified_jsma,
                    'success': True
                }
                print(f"    ✓ JSMA: {len(modified_jsma)} pixels modified")
        except Exception as e:
            print(f"    ✗ JSMA failed: {e}")
        
        # Greedy
        try:
            success, adv_greedy, modified_greedy = greedy_attack(
                image.squeeze(0), label.item(), self.model,
                max_pixels=10, step_size=0.3
            )
            if success:
                results['Greedy'] = {
                    'image': adv_greedy.unsqueeze(0),
                    'modified_pixels': modified_greedy,
                    'success': True
                }
                print(f"    ✓ Greedy: {len(modified_greedy)} pixels modified")
        except Exception as e:
            print(f"    ✗ Greedy failed: {e}")
        
        # SparseFool
        try:
            success, adv_sf, modified_sf = sparsefool_attack(
                image.squeeze(0), label.item(), self.model,
                max_iterations=30
            )
            if success:
                results['SparseFool'] = {
                    'image': adv_sf.unsqueeze(0),
                    'modified_pixels': modified_sf,
                    'success': True
                }
                print(f"    ✓ SparseFool: {len(modified_sf)} pixels modified")
        except Exception as e:
            print(f"    ✗ SparseFool failed: {e}")
        
        return results
    
    def visualize_single_comparison(self, sample, adv_results, sample_id):
        """可视化单个样本的对比"""
        num_methods = len(adv_results)
        if num_methods == 0:
            return
        
        # 创建图表：原图 + 各方法的对抗图 + 差异图
        fig = plt.figure(figsize=(4 * (1 + num_methods * 2), 5))
        
        # 原图
        ax = plt.subplot(1, 1 + num_methods * 2, 1)
        orig_img = self.tensor_to_image(sample['image'])
        ax.imshow(orig_img)
        ax.set_title(f'Original\n{self.class_names[sample["label"]]}',
                    fontweight='bold', fontsize=12)
        ax.axis('off')
        
        # 各方法的对抗样本和差异
        col = 2
        for method_name, result in adv_results.items():
            # 对抗样本
            ax = plt.subplot(1, 1 + num_methods * 2, col)
            adv_img = self.tensor_to_image(result['image'])
            ax.imshow(adv_img)
            
            # 获取对抗预测
            with torch.no_grad():
                adv_output = self.model(result['image'].to(self.device))
                adv_pred = adv_output.argmax(dim=1).item()
            
            ax.set_title(f'{method_name}\n→ {self.class_names[adv_pred]}',
                        fontweight='bold', fontsize=12, color='red')
            ax.axis('off')
            col += 1
            
            # 差异图（放大10倍）
            ax = plt.subplot(1, 1 + num_methods * 2, col)
            diff = np.abs(adv_img - orig_img)
            diff_enhanced = np.clip(diff * 10, 0, 1)  # 放大10倍以便观察
            ax.imshow(diff_enhanced)
            ax.set_title(f'Difference ×10\n{len(result["modified_pixels"])} pixels',
                        fontweight='bold', fontsize=12)
            ax.axis('off')
            col += 1
        
        plt.tight_layout()
        filename = self.output_dir / f'comparison_sample_{sample_id}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.savefig(filename.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Saved: {filename}")
    
    def create_pixel_heatmap(self, adv_results, sample_id):
        """创建像素修改位置热图"""
        if len(adv_results) == 0:
            return
        
        fig, axes = plt.subplots(1, len(adv_results), figsize=(5 * len(adv_results), 5))
        
        if len(adv_results) == 1:
            axes = [axes]
        
        for ax, (method_name, result) in zip(axes, adv_results.items()):
            # 创建热图（32x32）
            heatmap = np.zeros((32, 32))
            
            for pixel_info in result['modified_pixels']:
                if isinstance(pixel_info, tuple) and len(pixel_info) >= 2:
                    y, x = pixel_info[0], pixel_info[1]
                    if 0 <= y < 32 and 0 <= x < 32:
                        heatmap[y, x] += 1
            
            # 绘制热图
            sns.heatmap(heatmap, cmap='Reds', cbar=True, square=True,
                       xticklabels=False, yticklabels=False, ax=ax,
                       vmin=0, vmax=heatmap.max() + 0.1)
            ax.set_title(f'{method_name}\n{len(result["modified_pixels"])} pixels',
                        fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        filename = self.output_dir / f'heatmap_sample_{sample_id}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.savefig(filename.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Saved heatmap: {filename}")
    
    def create_aggregated_heatmap(self, all_results):
        """创建所有样本的聚合热图"""
        print("\nCreating aggregated heatmaps...")
        
        # 为每个方法创建聚合热图
        method_heatmaps = {}
        
        for sample_results in all_results:
            for method_name, result in sample_results.items():
                if method_name not in method_heatmaps:
                    method_heatmaps[method_name] = np.zeros((32, 32))
                
                for pixel_info in result['modified_pixels']:
                    if isinstance(pixel_info, tuple) and len(pixel_info) >= 2:
                        y, x = pixel_info[0], pixel_info[1]
                        if 0 <= y < 32 and 0 <= x < 32:
                            method_heatmaps[method_name][y, x] += 1
        
        # 绘制聚合热图
        num_methods = len(method_heatmaps)
        fig, axes = plt.subplots(1, num_methods, figsize=(6 * num_methods, 6))
        
        if num_methods == 1:
            axes = [axes]
        
        for ax, (method_name, heatmap) in zip(axes, method_heatmaps.items()):
            sns.heatmap(heatmap, cmap='YlOrRd', cbar=True, square=True,
                       xticklabels=False, yticklabels=False, ax=ax)
            total_modifications = int(heatmap.sum())
            ax.set_title(f'{method_name}\nTotal: {total_modifications} modifications',
                        fontweight='bold', fontsize=14)
        
        plt.suptitle('Pixel Modification Patterns Across All Samples',
                    fontweight='bold', fontsize=16, y=1.02)
        plt.tight_layout()
        
        filename = self.output_dir / 'aggregated_heatmap.png'
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.savefig(filename.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved aggregated heatmap: {filename}")
    
    def create_grid_comparison(self, samples, all_results):
        """创建网格对比图（所有样本一起）"""
        print("\nCreating grid comparison...")
        
        num_samples = len(samples)
        num_methods = len(all_results[0]) if all_results else 0
        
        if num_methods == 0:
            return
        
        # 创建网格：行=样本，列=原图+方法
        fig = plt.figure(figsize=(3 * (1 + num_methods), 3 * num_samples))
        
        for i, (sample, adv_results) in enumerate(zip(samples, all_results)):
            # 原图
            ax = plt.subplot(num_samples, 1 + num_methods, i * (1 + num_methods) + 1)
            orig_img = self.tensor_to_image(sample['image'])
            ax.imshow(orig_img)
            if i == 0:
                ax.set_title('Original', fontweight='bold', fontsize=12)
            ax.set_ylabel(f'Sample {i+1}', fontweight='bold', rotation=0, labelpad=40)
            ax.axis('off')
            
            # 各方法
            for j, (method_name, result) in enumerate(adv_results.items(), 1):
                ax = plt.subplot(num_samples, 1 + num_methods, i * (1 + num_methods) + 1 + j)
                adv_img = self.tensor_to_image(result['image'])
                ax.imshow(adv_img)
                
                if i == 0:
                    ax.set_title(method_name, fontweight='bold', fontsize=12)
                
                ax.axis('off')
        
        plt.suptitle('Adversarial Examples Comparison', fontweight='bold', fontsize=16, y=0.995)
        plt.tight_layout()
        
        filename = self.output_dir / 'grid_comparison.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.savefig(filename.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved grid comparison: {filename}")
    
    def run_visualization(self, num_samples=5):
        """运行完整可视化"""
        print("\n" + "🚀"*30)
        print("对抗样本可视化")
        print("🚀"*30)
        
        # 1. 选择样本
        samples = self.select_samples_for_visualization(num_samples)
        
        # 2. 生成对抗样本
        all_results = []
        for i, sample in enumerate(samples, 1):
            print(f"\nProcessing sample {i}/{num_samples}...")
            adv_results = self.generate_adversarial_examples(sample)
            all_results.append(adv_results)
            
            # 3. 单个样本对比图
            self.visualize_single_comparison(sample, adv_results, i)
            
            # 4. 像素热图
            self.create_pixel_heatmap(adv_results, i)
        
        # 5. 聚合热图
        self.create_aggregated_heatmap(all_results)
        
        # 6. 网格对比
        self.create_grid_comparison(samples, all_results)
        
        # 完成
        print("\n" + "🎉"*30)
        print("可视化完成！")
        print("🎉"*30)
        
        print(f"\n📁 生成的文件：")
        print(f"  • {num_samples} 个样本对比图")
        print(f"  • {num_samples} 个像素热图")
        print(f"  • 1 个聚合热图")
        print(f"  • 1 个网格对比图")
        print(f"  • 总计：{(num_samples * 2 + 2) * 2} 个文件（PNG + PDF）")
        
        print(f"\n📂 保存位置：{self.output_dir}")
        
        return all_results

def main():
    visualizer = AdversarialVisualizer()
    visualizer.run_visualization(num_samples=5)
    return 0

if __name__ == '__main__':
    exit(main())

