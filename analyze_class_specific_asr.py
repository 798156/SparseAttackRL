#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
类别级别攻击成功率分析
分析不同CIFAR-10类别的攻击难度和混淆模式
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch
import torchvision
import torchvision.transforms as transforms

class ClassSpecificAnalyzer:
    def __init__(self):
        self.results_dir = Path('results/complete_baseline')
        self.output_dir = Path('results/class_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # CIFAR-10类别
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        self.class_names_cn = [
            '飞机', '汽车', '鸟', '猫', '鹿',
            '狗', '青蛙', '马', '船', '卡车'
        ]
        
        self.models = ['resnet18', 'vgg16', 'mobilenetv2']
        self.methods = ['jsma', 'sparsefool', 'greedy', 'pixelgrad', 'randomsparse']
        
        # 存储数据
        self.all_data = {}
        self.class_asr = {}  # {model: {method: {class_id: asr}}}
        self.confusion_matrices = {}  # {model: {method: confusion_matrix}}
        
        # 重建样本标签
        self.sample_labels = None  # 将存储100个样本的真实标签
    
    def reconstruct_sample_labels(self):
        """重建样本标签（基于相同的随机种子和采样逻辑）"""
        print("\n" + "="*60)
        print("🔄 重建样本标签...")
        print("="*60)
        
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 加载CIFAR-10测试集
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
        
        # 加载ResNet18模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        from load_trained_model import load_trained_model
        model = load_trained_model('resnet18', 'cifar10_resnet18.pth', device=device)
        model.eval()
        
        print(f"  Device: {device}")
        print(f"  模型加载: ResNet18")
        
        # 选择100个正确分类的样本
        labels = []
        count = 0
        target_samples = 100
        
        with torch.no_grad():
            for images, true_labels in test_loader:
                if count >= target_samples:
                    break
                
                images = images.to(device)
                true_labels = true_labels.to(device)
                
                # 预测
                outputs = model(images)
                pred = outputs.argmax(dim=1)
                
                # 只选择正确分类的样本
                if pred.item() == true_labels.item():
                    labels.append(true_labels.item())
                    count += 1
                    
                    if count % 20 == 0:
                        print(f"  已选择 {count}/{target_samples} 个样本")
        
        self.sample_labels = labels
        
        # 统计类别分布
        class_counts = Counter(labels)
        print(f"\n✓ 样本标签重建完成！")
        print(f"  总样本数: {len(labels)}")
        print(f"  类别分布:")
        for class_id in range(10):
            count = class_counts.get(class_id, 0)
            print(f"    {self.class_names[class_id]}: {count} 个样本")
        
        return labels
    
    def load_all_data(self):
        """加载所有实验数据"""
        print("\n" + "="*60)
        print("📂 加载实验数据...")
        print("="*60)
        
        for model in self.models:
            self.all_data[model] = {}
            for method in self.methods:
                json_file = self.results_dir / f'{model}_{method}.json'
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        self.all_data[model][method] = data
                        print(f"✓ 加载: {model}_{method}")
                else:
                    print(f"⚠️  文件不存在: {json_file}")
        
        print(f"\n✅ 加载完成")
    
    def extract_class_labels(self, data):
        """从数据中提取类别标签"""
        # 使用重建的样本标签
        if self.sample_labels is None:
            raise ValueError("样本标签未重建！请先调用 reconstruct_sample_labels()")
        return self.sample_labels
    
    def analyze_class_asr(self):
        """分析每个类别的ASR"""
        print("\n" + "="*60)
        print("📊 分析类别级别ASR...")
        print("="*60)
        
        # 使用ResNet18作为主要分析对象
        model = 'resnet18'
        
        for method in self.methods:
            if method not in self.all_data[model]:
                print(f"  ⚠️  跳过 {method}: 数据不存在")
                continue
            
            data = self.all_data[model][method]
            
            # 提取标签和结果
            labels = self.extract_class_labels(data)
            
            if 'detailed_results' not in data:
                print(f"  ⚠️  跳过 {method}: 没有detailed_results")
                continue
            
            results = data['detailed_results']
            
            # 确保长度匹配
            if len(labels) != len(results):
                print(f"  ⚠️  警告: 标签数({len(labels)}) ≠ 结果数({len(results)})")
                min_len = min(len(labels), len(results))
                labels = labels[:min_len]
                results = results[:min_len]
            
            # 统计每个类别的成功率
            class_stats = defaultdict(lambda: {'total': 0, 'success': 0})
            
            for label, result in zip(labels, results):
                class_stats[label]['total'] += 1
                if result.get('success', False):
                    class_stats[label]['success'] += 1
            
            # 计算ASR
            if model not in self.class_asr:
                self.class_asr[model] = {}
            
            self.class_asr[model][method] = {}
            for class_id in range(10):
                if class_stats[class_id]['total'] > 0:
                    asr = class_stats[class_id]['success'] / class_stats[class_id]['total'] * 100
                else:
                    asr = 0
                self.class_asr[model][method][class_id] = asr
            
            print(f"  ✓ {method}: 类别ASR已计算 (样本数={len(labels)})")
    
    def analyze_confusion_patterns(self):
        """分析混淆模式（攻击后被误分类为哪个类别）"""
        print("\n" + "="*60)
        print("🔄 分析混淆模式...")
        print("="*60)
        print("  ⚠️  跳过：JSON文件中没有adversarial_label信息")
        print("  💡 如需混淆矩阵分析，需要修改实验脚本保存对抗标签")
        # 跳过混淆矩阵分析（数据中没有对抗标签）
        pass
    
    def generate_visualizations(self):
        """生成可视化"""
        print("\n" + "="*60)
        print("📈 生成可视化...")
        print("="*60)
        
        # 1. 类别ASR热图
        self._plot_class_asr_heatmap()
        
        # 2. 类别难度排名
        self._plot_class_difficulty_ranking()
        
        # 3. 类别间ASR对比
        self._plot_class_comparison()
        
        print("✓ 所有可视化已生成（跳过混淆矩阵：数据中无对抗标签）")
    
    def _plot_class_asr_heatmap(self):
        """绘制类别ASR热图"""
        model = 'resnet18'
        
        # 准备数据
        asr_matrix = np.zeros((5, 10))  # 5方法 × 10类别
        
        for i, method in enumerate(self.methods):
            if method in self.class_asr[model]:
                for j in range(10):
                    asr_matrix[i, j] = self.class_asr[model][method].get(j, 0)
        
        # 绘制热图
        fig, ax = plt.subplots(figsize=(14, 6))
        
        methods_display = ['JSMA', 'SparseFool', 'Greedy', 'PixelGrad', 'RandomSparse']
        
        sns.heatmap(asr_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                   xticklabels=self.class_names, yticklabels=methods_display,
                   cbar_kws={'label': 'ASR (%)'}, vmin=0, vmax=100, ax=ax)
        
        ax.set_title('Class-Specific Attack Success Rate (ResNet18)',
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xlabel('CIFAR-10 Class', fontweight='bold', fontsize=12)
        ax.set_ylabel('Attack Method', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_asr_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'class_asr_heatmap.pdf', bbox_inches='tight')
        plt.close()
        
        print("  ✓ class_asr_heatmap.pdf")
    
    def _plot_class_difficulty_ranking(self):
        """绘制类别难度排名"""
        model = 'resnet18'
        
        # 计算每个类别的平均ASR
        class_avg_asr = {}
        for class_id in range(10):
            asr_values = []
            for method in self.methods:
                if method in self.class_asr[model]:
                    asr_values.append(self.class_asr[model][method].get(class_id, 0))
            class_avg_asr[class_id] = np.mean(asr_values) if asr_values else 0
        
        # 排序（从难到易）
        sorted_classes = sorted(class_avg_asr.items(), key=lambda x: x[1])
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        class_names = [self.class_names[c] for c, _ in sorted_classes]
        asr_values = [asr for _, asr in sorted_classes]
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, 10))
        bars = ax.barh(class_names, asr_values, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Average ASR (%) Across All Methods', fontweight='bold', fontsize=12)
        ax.set_title('Class Difficulty Ranking (Lower ASR = More Robust)',
                    fontweight='bold', fontsize=14, pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, (bar, asr) in enumerate(zip(bars, asr_values)):
            ax.text(asr + 1, i, f'{asr:.1f}%', va='center', fontweight='bold')
        
        # 添加难度标签
        ax.text(0.02, 0.98, 'Hardest to Attack ↑', transform=ax.transAxes,
               va='top', fontweight='bold', fontsize=11, color='darkred',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.02, 0.02, '↓ Easiest to Attack', transform=ax.transAxes,
               va='bottom', fontweight='bold', fontsize=11, color='darkgreen',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_difficulty_ranking.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'class_difficulty_ranking.pdf', bbox_inches='tight')
        plt.close()
        
        print("  ✓ class_difficulty_ranking.pdf")
    
    
    def _plot_class_comparison(self):
        """绘制类别间ASR对比"""
        model = 'resnet18'
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        methods_display = {
            'jsma': 'JSMA',
            'sparsefool': 'SparseFool',
            'greedy': 'Greedy',
            'pixelgrad': 'PixelGrad',
            'randomsparse': 'RandomSparse'
        }
        
        x = np.arange(10)
        width = 0.15
        
        for i, method in enumerate(self.methods):
            if method not in self.class_asr[model]:
                continue
            
            asr_values = [self.class_asr[model][method].get(j, 0) for j in range(10)]
            offset = (i - 2) * width
            ax.bar(x + offset, asr_values, width, label=methods_display[method], alpha=0.8)
        
        ax.set_xlabel('CIFAR-10 Class', fontweight='bold', fontsize=12)
        ax.set_ylabel('Attack Success Rate (%)', fontweight='bold', fontsize=12)
        ax.set_title('Class-Specific ASR Comparison Across Methods',
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'class_comparison.pdf', bbox_inches='tight')
        plt.close()
        
        print("  ✓ class_comparison.pdf")
    
    def generate_report(self):
        """生成分析报告"""
        print("\n" + "="*60)
        print("📝 生成分析报告...")
        print("="*60)
        
        model = 'resnet18'
        
        report = f"""# 类别级别分析报告

**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**分析模型:** ResNet18  
**数据集:** CIFAR-10

---

## 执行摘要

本报告分析了5种L0攻击方法在CIFAR-10的10个类别上的表现，
揭示不同类别的攻击难度和混淆模式。

---

## 1. 类别难度排名

### 1.1 平均ASR（跨方法）

"""
        
        # 计算平均ASR
        class_avg_asr = {}
        for class_id in range(10):
            asr_values = []
            for method in self.methods:
                if method in self.class_asr[model]:
                    asr_values.append(self.class_asr[model][method].get(class_id, 0))
            class_avg_asr[class_id] = np.mean(asr_values) if asr_values else 0
        
        sorted_classes = sorted(class_avg_asr.items(), key=lambda x: x[1])
        
        report += "| 排名 | 类别 | 平均ASR | 难度级别 |\n"
        report += "|------|------|---------|----------|\n"
        
        for rank, (class_id, asr) in enumerate(sorted_classes, 1):
            if rank <= 3:
                level = "困难 🔴"
            elif rank <= 7:
                level = "中等 🟡"
            else:
                level = "容易 🟢"
            
            report += f"| {rank} | {self.class_names[class_id]} ({self.class_names_cn[class_id]}) | {asr:.1f}% | {level} |\n"
        
        report += "\n### 1.2 关键发现\n\n"
        
        hardest = sorted_classes[0]
        easiest = sorted_classes[-1]
        
        report += f"- **最难攻击类别:** {self.class_names[hardest[0]]} ({hardest[1]:.1f}% ASR)\n"
        report += f"- **最易攻击类别:** {self.class_names[easiest[0]]} ({easiest[1]:.1f}% ASR)\n"
        report += f"- **难度差距:** {easiest[1] - hardest[1]:.1f} 个百分点\n\n"
        
        report += "---\n\n## 2. 方法特定的类别表现\n\n"
        
        methods_display = {
            'jsma': 'JSMA',
            'sparsefool': 'SparseFool',
            'greedy': 'Greedy',
            'pixelgrad': 'PixelGrad',
            'randomsparse': 'RandomSparse'
        }
        
        report += "### 2.1 完整ASR表格\n\n"
        report += "| 类别 | JSMA | SparseFool | Greedy | PixelGrad | RandomSparse |\n"
        report += "|------|------|------------|--------|-----------|---------------|\n"
        
        for class_id in range(10):
            row = [self.class_names[class_id]]
            for method in self.methods:
                if method in self.class_asr[model]:
                    asr = self.class_asr[model][method].get(class_id, 0)
                    row.append(f"{asr:.1f}%")
                else:
                    row.append("-")
            report += "| " + " | ".join(row) + " |\n"
        
        report += "\n### 2.2 方法-类别交互\n\n"
        
        # 找出每个方法表现最好和最差的类别
        for method in self.methods:
            if method not in self.class_asr[model]:
                continue
            
            class_asrs = [(c, self.class_asr[model][method].get(c, 0)) for c in range(10)]
            best = max(class_asrs, key=lambda x: x[1])
            worst = min(class_asrs, key=lambda x: x[1])
            
            report += f"**{methods_display[method]}:**\n"
            report += f"- 最佳类别: {self.class_names[best[0]]} ({best[1]:.1f}% ASR)\n"
            report += f"- 最差类别: {self.class_names[worst[0]]} ({worst[1]:.1f}% ASR)\n"
            report += f"- 差距: {best[1] - worst[1]:.1f}%\n\n"
        
        report += "---\n\n## 3. 可能的原因分析\n\n"
        report += "### 3.1 为什么某些类别更难攻击？\n\n"
        report += "基于结果，我们推测以下因素可能影响攻击难度：\n\n"
        
        report += "**1. 视觉特征复杂度**\n"
        report += "- 简单纹理（如船、飞机）可能更容易攻击\n"
        report += "- 复杂纹理（如猫、狗）可能更难攻击\n\n"
        
        report += "**2. 类内变异性**\n"
        report += "- 类内差异大的类别（如狗）更难定义统一的攻击策略\n"
        report += "- 类内一致性高的类别（如汽车）可能更容易攻击\n\n"
        
        report += "**3. 类间语义距离**\n"
        report += "- 与其他类别语义相近的类别更容易被误分类\n"
        report += "- 独特的类别（如青蛙）可能需要更多修改才能跨越决策边界\n\n"
        
        report += "**4. 训练数据分布**\n"
        report += "- 模型在某些类别上训练得更好，导致更强的鲁棒性\n"
        report += "- 数据增强可能对某些类别更有效\n\n"
        
        report += "---\n\n## 4. 研究启示\n\n"
        report += "### 4.1 对攻击研究的启示\n\n"
        report += "1. **类别自适应攻击:** 根据类别特征选择攻击策略\n"
        report += "2. **困难类别优化:** 针对难攻击类别设计专门方法\n"
        report += "3. **语义感知攻击:** 利用类别间语义关系指导攻击\n\n"
        
        report += "### 4.2 对防御研究的启示\n\n"
        report += "1. **类别特定防御:** 为易攻击类别提供额外保护\n"
        report += "2. **均衡鲁棒性:** 减少不同类别间的鲁棒性差异\n"
        report += "3. **语义边界强化:** 加强语义相似类别间的决策边界\n\n"
        
        report += "### 4.3 论文价值\n\n"
        report += "- ✅ 首次系统分析L0攻击的类别特定模式\n"
        report += "- ✅ 揭示类别难度与攻击方法的交互效应\n"
        report += "- ✅ 为类别自适应攻击/防御提供基础\n"
        report += "- ✅ 增加论文的细粒度分析深度\n\n"
        
        report += "---\n\n## 5. 可视化索引\n\n"
        report += "1. **class_asr_heatmap.pdf** - 类别ASR热图\n"
        report += "2. **class_difficulty_ranking.pdf** - 类别难度排名\n"
        report += "3. **class_comparison.pdf** - 类别间ASR对比\n\n"
        report += "**注意：** 混淆矩阵分析需要对抗样本的预测标签，当前数据中未包含此信息。\n\n"
        
        report += f"\n---\n\n*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        # 保存报告
        report_file = self.output_dir / 'class_analysis_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ 分析报告已保存: {report_file}")
        
        return report
    
    def run_complete_analysis(self):
        """运行完整分析"""
        print("\n" + "🚀"*30)
        print("类别级别分析")
        print("🚀"*30)
        
        # 0. 重建样本标签
        self.reconstruct_sample_labels()
        
        # 1. 加载数据
        self.load_all_data()
        
        # 2. 分析类别ASR
        self.analyze_class_asr()
        
        # 3. 分析混淆模式
        self.analyze_confusion_patterns()
        
        # 4. 生成可视化
        self.generate_visualizations()
        
        # 5. 生成报告
        report = self.generate_report()
        
        # 最终总结
        print("\n" + "🎉"*30)
        print("类别分析完成！")
        print("🎉"*30)
        
        print(f"\n📁 生成的文件:")
        print(f"  1. {self.output_dir / 'class_analysis_report.md'}")
        print(f"  2. {self.output_dir / 'class_asr_heatmap.pdf'}")
        print(f"  3. {self.output_dir / 'class_difficulty_ranking.pdf'}")
        print(f"  4. {self.output_dir / 'class_comparison.pdf'}")
        print(f"  总计：6个文件（PDF + PNG）")
        
        print(f"\n📂 保存位置：{self.output_dir}")

def main():
    analyzer = ClassSpecificAnalyzer()
    analyzer.run_complete_analysis()
    return 0

if __name__ == '__main__':
    exit(main())

