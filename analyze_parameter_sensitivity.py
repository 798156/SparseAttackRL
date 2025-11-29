#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数敏感性分析
测试不同参数对攻击性能的影响
目标：展示方法的稳定性，找出最优参数
"""

import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 非GUI后端
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 导入攻击方法适配器
from attack_adapters import (
    jsma_attack_adapter,
    sparsefool_attack_adapter,
    greedy_attack_adapter,
    pixel_gradient_attack_adapter,
    random_sparse_attack_adapter
)

# 导入模型和数据加载器
from load_trained_model import load_trained_model
from dataset_loader import DatasetLoader

# 配置
CONFIG = {
    'model_name': 'resnet18',  # 使用ResNet18作为代表性模型
    'model_path': 'cifar10_resnet18.pth',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'samples_per_config': 50,  # 每个参数配置50个样本
    'random_seed': 42,
    'output_dir': Path('results/parameter_sensitivity')
}

# 参数配置
PARAMETER_CONFIGS = {
    'JSMA': {
        'param_name': 'max_pixels',
        'param_values': [5, 10, 15, 20],
        'fixed_params': {'theta': 0.2, 'max_iterations': 100}
    },
    'SparseFool': {
        'param_name': 'max_iter',
        'param_values': [10, 20, 30, 50],
        'fixed_params': {'overshoot': 0.02, 'lambda_': 3.0}
    },
    'Greedy': {
        'param_name': 'max_pixels',
        'param_values': [5, 10, 15, 20],
        'fixed_params': {'alpha': 0.1, 'max_iterations': 100}
    },
    'PixelGrad': {
        'param_name': 'max_pixels',
        'param_values': [5, 10, 15, 20],
        'fixed_params': {'alpha': 0.2, 'beta': 0.9}
    },
    'RandomSparse': {
        'param_name': 'max_pixels',
        'param_values': [5, 10, 15, 20],
        'fixed_params': {'perturbation_size': 0.2, 'max_attempts': 50}
    }
}

class ParameterSensitivityAnalyzer:
    def __init__(self):
        self.device = CONFIG['device']
        self.output_dir = CONFIG['output_dir']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载模型
        print(f"Loading model: {CONFIG['model_name']}")
        self.model = load_trained_model(
            CONFIG['model_name'], 
            CONFIG['model_path'], 
            device=self.device,
            num_classes=10
        )
        self.model.eval()
        
        # 加载测试数据
        print("Loading test data...")
        dataset_loader = DatasetLoader(dataset_name='cifar10', data_root='./data')
        test_loader = dataset_loader.load_test_set()
        
        # 选择测试样本（只选择分类正确的）
        print(f"Selecting {CONFIG['samples_per_config']} correctly classified samples...")
        self.test_samples = self._select_correct_samples(test_loader, CONFIG['samples_per_config'])
        print(f"✓ Selected {len(self.test_samples)} samples")
        
        # 存储结果
        self.results = {}
        
    def _select_correct_samples(self, test_loader, num_samples):
        """选择分类正确的样本"""
        samples = []
        with torch.no_grad():
            for images, labels in test_loader:
                if len(samples) >= num_samples:
                    break
                
                # 确保labels是tensor
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor([labels])
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 确保images有batch维度
                if images.dim() == 3:
                    images = images.unsqueeze(0)
                    
                outputs = self.model(images)
                pred = outputs.argmax(dim=1)
                
                if pred.item() == labels.item():
                    samples.append({
                        'image': images.cpu(),
                        'label': labels.cpu(),
                        'pred': pred.cpu()
                    })
        return samples
    
    def test_configuration(self, method_name, param_name, param_value, fixed_params):
        """测试一个参数配置"""
        print(f"\n{'='*60}")
        print(f"Testing {method_name}: {param_name}={param_value}")
        print(f"{'='*60}")
        
        results = []
        successful_attacks = 0
        
        for idx, sample in enumerate(tqdm(self.test_samples, desc=f"{method_name}")):
            image = sample['image'].to(self.device)
            label = sample['label'].to(self.device)
            
            # 准备参数
            params = fixed_params.copy()
            params[param_name] = param_value
            
            try:
                # 调用攻击方法适配器
                if method_name == 'JSMA':
                    adv_image, success, info = jsma_attack_adapter(
                        self.model, image, label,
                        max_pixels=params['max_pixels'],
                        theta=params['theta'],
                        max_iterations=params['max_iterations'],
                        device=self.device
                    )
                elif method_name == 'SparseFool':
                    adv_image, success, info = sparsefool_attack_adapter(
                        self.model, image, label,
                        max_iter=params['max_iter'],
                        overshoot=params['overshoot'],
                        lambda_=params['lambda_'],
                        device=self.device
                    )
                elif method_name == 'Greedy':
                    adv_image, success, info = greedy_attack_adapter(
                        self.model, image, label,
                        max_pixels=params['max_pixels'],
                        alpha=params['alpha'],
                        max_iterations=params['max_iterations'],
                        device=self.device
                    )
                elif method_name == 'PixelGrad':
                    adv_image, success, info = pixel_gradient_attack_adapter(
                        self.model, image, label,
                        max_pixels=params['max_pixels'],
                        alpha=params['alpha'],
                        beta=params['beta'],
                        device=self.device
                    )
                elif method_name == 'RandomSparse':
                    adv_image, success, info = random_sparse_attack_adapter(
                        self.model, image, label,
                        max_pixels=params['max_pixels'],
                        perturbation_size=params['perturbation_size'],
                        max_attempts=params['max_attempts'],
                        device=self.device
                    )
                else:
                    raise ValueError(f"Unknown method: {method_name}")
                
                if success:
                    successful_attacks += 1
                
                results.append({
                    'sample_idx': idx,
                    'success': success,
                    'l0_norm': info.get('l0_norm', 0),
                    'l2_norm': info.get('l2_norm', 0),
                    'time': info.get('time', 0),
                    'iterations': info.get('iterations', 0)
                })
                
            except Exception as e:
                print(f"⚠️ Sample {idx} error: {e}")
                results.append({
                    'sample_idx': idx,
                    'success': False,
                    'error': str(e)
                })
        
        # 计算统计信息
        asr = (successful_attacks / len(self.test_samples)) * 100
        successful_results = [r for r in results if r.get('success', False)]
        
        if successful_results:
            avg_l0 = np.mean([r['l0_norm'] for r in successful_results])
            avg_l2 = np.mean([r['l2_norm'] for r in successful_results])
            avg_time = np.mean([r['time'] for r in successful_results])
            avg_iterations = np.mean([r['iterations'] for r in successful_results])
        else:
            avg_l0 = avg_l2 = avg_time = avg_iterations = 0
        
        summary = {
            'method': method_name,
            'param_name': param_name,
            'param_value': param_value,
            'fixed_params': fixed_params,
            'asr': round(asr, 1),
            'avg_l0': round(avg_l0, 2),
            'avg_l2': round(avg_l2, 4),
            'avg_time': round(avg_time, 3),
            'avg_iterations': round(avg_iterations, 1),
            'num_samples': len(self.test_samples),
            'successful_attacks': successful_attacks
        }
        
        print(f"\n📊 Results: ASR={asr:.1f}%, L0={avg_l0:.2f}, Time={avg_time:.3f}s")
        
        return summary, results
    
    def run_sensitivity_analysis(self):
        """运行完整的敏感性分析"""
        print(f"\n{'🚀'*30}")
        print("Starting Parameter Sensitivity Analysis")
        print(f"{'🚀'*30}")
        print(f"\nDevice: {self.device}")
        print(f"Model: {CONFIG['model_name']}")
        print(f"Samples per config: {CONFIG['samples_per_config']}")
        
        # 测试所有方法的所有参数配置
        for method_name, config in PARAMETER_CONFIGS.items():
            print(f"\n\n{'#'*60}")
            print(f"# Method: {method_name}")
            print(f"# Parameter: {config['param_name']}")
            print(f"# Values: {config['param_values']}")
            print(f"{'#'*60}")
            
            method_results = []
            
            for param_value in config['param_values']:
                summary, detailed_results = self.test_configuration(
                    method_name,
                    config['param_name'],
                    param_value,
                    config['fixed_params']
                )
                method_results.append({
                    'summary': summary,
                    'detailed': detailed_results
                })
            
            # 保存每个方法的结果
            self.results[method_name] = method_results
            self._save_method_results(method_name, method_results)
        
        # 生成分析报告和图表
        self.generate_visualizations()
        self.generate_report()
        
        print(f"\n\n{'🎉'*30}")
        print("Parameter Sensitivity Analysis Completed!")
        print(f"{'🎉'*30}")
        print(f"\n✅ All results saved to: {self.output_dir}")
    
    def _save_method_results(self, method_name, results):
        """保存单个方法的结果"""
        # 转换numpy类型为Python原生类型
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj
        
        results_native = convert_to_native(results)
        
        filename = self.output_dir / f'{method_name.lower()}_sensitivity.json'
        with open(filename, 'w') as f:
            json.dump(results_native, f, indent=2)
        print(f"✓ Saved: {filename}")
    
    def generate_visualizations(self):
        """生成可视化图表"""
        print(f"\n{'='*60}")
        print("Generating Visualizations...")
        print(f"{'='*60}")
        
        # 1. ASR vs Parameter 曲线图
        self._plot_asr_curves()
        
        # 2. L0 Norm vs Parameter 曲线图
        self._plot_l0_curves()
        
        # 3. 效率对比（Time vs Parameter）
        self._plot_time_curves()
        
        # 4. 综合对比热图
        self._plot_comprehensive_heatmap()
        
        print("✓ All visualizations generated")
    
    def _plot_asr_curves(self):
        """绘制ASR敏感性曲线"""
        plt.figure(figsize=(12, 7))
        
        for method_name, method_results in self.results.items():
            config = PARAMETER_CONFIGS[method_name]
            param_values = [r['summary']['param_value'] for r in method_results]
            asr_values = [r['summary']['asr'] for r in method_results]
            
            plt.plot(param_values, asr_values, marker='o', linewidth=2, 
                    markersize=8, label=method_name)
        
        plt.xlabel('Parameter Value', fontsize=12, fontweight='bold')
        plt.ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
        plt.title('Parameter Sensitivity: ASR vs Parameter Value', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存
        plt.savefig(self.output_dir / 'asr_sensitivity_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'asr_sensitivity_curves.pdf', bbox_inches='tight')
        plt.close()
        print("  ✓ ASR sensitivity curves")
    
    def _plot_l0_curves(self):
        """绘制L0 Norm敏感性曲线"""
        plt.figure(figsize=(12, 7))
        
        for method_name, method_results in self.results.items():
            param_values = [r['summary']['param_value'] for r in method_results]
            l0_values = [r['summary']['avg_l0'] for r in method_results]
            
            plt.plot(param_values, l0_values, marker='s', linewidth=2,
                    markersize=8, label=method_name)
        
        plt.xlabel('Parameter Value', fontsize=12, fontweight='bold')
        plt.ylabel('Average L0 Norm (pixels)', fontsize=12, fontweight='bold')
        plt.title('Parameter Sensitivity: L0 Norm vs Parameter Value',
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'l0_sensitivity_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'l0_sensitivity_curves.pdf', bbox_inches='tight')
        plt.close()
        print("  ✓ L0 sensitivity curves")
    
    def _plot_time_curves(self):
        """绘制时间效率曲线"""
        plt.figure(figsize=(12, 7))
        
        for method_name, method_results in self.results.items():
            param_values = [r['summary']['param_value'] for r in method_results]
            time_values = [r['summary']['avg_time'] for r in method_results]
            
            plt.plot(param_values, time_values, marker='^', linewidth=2,
                    markersize=8, label=method_name)
        
        plt.xlabel('Parameter Value', fontsize=12, fontweight='bold')
        plt.ylabel('Average Time (seconds)', fontsize=12, fontweight='bold')
        plt.title('Parameter Sensitivity: Time vs Parameter Value',
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'time_sensitivity_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'time_sensitivity_curves.pdf', bbox_inches='tight')
        plt.close()
        print("  ✓ Time sensitivity curves")
    
    def _plot_comprehensive_heatmap(self):
        """绘制综合对比热图"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 准备数据
        methods = list(self.results.keys())
        max_params = max([len(PARAMETER_CONFIGS[m]['param_values']) for m in methods])
        
        asr_matrix = np.zeros((len(methods), max_params))
        l0_matrix = np.zeros((len(methods), max_params))
        time_matrix = np.zeros((len(methods), max_params))
        
        param_labels = []
        for i, method_name in enumerate(methods):
            results = self.results[method_name]
            param_values = PARAMETER_CONFIGS[method_name]['param_values']
            
            for j, result in enumerate(results):
                asr_matrix[i, j] = result['summary']['asr']
                l0_matrix[i, j] = result['summary']['avg_l0']
                time_matrix[i, j] = result['summary']['avg_time']
            
            if i == 0:
                param_labels = [str(v) for v in param_values]
        
        # ASR 热图
        sns.heatmap(asr_matrix[:, :len(param_labels)], annot=True, fmt='.1f', 
                   cmap='RdYlGn', ax=axes[0], cbar_kws={'label': 'ASR (%)'},
                   xticklabels=param_labels, yticklabels=methods)
        axes[0].set_title('Attack Success Rate', fontweight='bold', fontsize=12)
        axes[0].set_xlabel('Parameter Value', fontweight='bold')
        
        # L0 热图
        sns.heatmap(l0_matrix[:, :len(param_labels)], annot=True, fmt='.2f',
                   cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'L0 Norm'},
                   xticklabels=param_labels, yticklabels=methods)
        axes[1].set_title('Average L0 Norm', fontweight='bold', fontsize=12)
        axes[1].set_xlabel('Parameter Value', fontweight='bold')
        
        # Time 热图
        sns.heatmap(time_matrix[:, :len(param_labels)], annot=True, fmt='.3f',
                   cmap='Blues', ax=axes[2], cbar_kws={'label': 'Time (s)'},
                   xticklabels=param_labels, yticklabels=methods)
        axes[2].set_title('Average Time', fontweight='bold', fontsize=12)
        axes[2].set_xlabel('Parameter Value', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'comprehensive_heatmap.pdf', bbox_inches='tight')
        plt.close()
        print("  ✓ Comprehensive heatmap")
    
    def generate_report(self):
        """生成分析报告"""
        print(f"\n{'='*60}")
        print("Generating Analysis Report...")
        print(f"{'='*60}")
        
        report = f"""# Parameter Sensitivity Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Model:** {CONFIG['model_name']}  
**Samples per configuration:** {CONFIG['samples_per_config']}

---

## Executive Summary

This report analyzes the sensitivity of 5 sparse adversarial attack methods to their key parameters.
The goal is to identify optimal parameters and assess method stability.

---

## Methods and Parameters Tested

"""
        
        for method_name, config in PARAMETER_CONFIGS.items():
            report += f"### {method_name}\n"
            report += f"- **Parameter:** {config['param_name']}\n"
            report += f"- **Values:** {config['param_values']}\n"
            report += f"- **Fixed params:** {config['fixed_params']}\n\n"
        
        report += "---\n\n## Detailed Results\n\n"
        
        # 为每个方法生成详细表格
        for method_name, method_results in self.results.items():
            report += f"### {method_name}\n\n"
            report += "| Parameter | ASR (%) | Avg L0 | Avg L2 | Avg Time (s) |\n"
            report += "|-----------|---------|--------|--------|-------------|\n"
            
            for result in method_results:
                s = result['summary']
                report += f"| {s['param_value']} | {s['asr']:.1f} | {s['avg_l0']:.2f} | "
                report += f"{s['avg_l2']:.4f} | {s['avg_time']:.3f} |\n"
            
            report += "\n"
        
        report += "---\n\n## Key Findings\n\n"
        
        # 找出最优参数
        report += "### Optimal Parameters\n\n"
        for method_name, method_results in self.results.items():
            # 找最高ASR
            best_asr = max(method_results, key=lambda x: x['summary']['asr'])
            param_name = PARAMETER_CONFIGS[method_name]['param_name']
            report += f"- **{method_name}:** {param_name}={best_asr['summary']['param_value']} "
            report += f"(ASR={best_asr['summary']['asr']:.1f}%)\n"
        
        report += "\n### Stability Ranking\n\n"
        report += "Methods ranked by parameter sensitivity (lower variance = more stable):\n\n"
        
        # 计算方差
        stability_scores = {}
        for method_name, method_results in self.results.items():
            asr_values = [r['summary']['asr'] for r in method_results]
            variance = np.var(asr_values)
            std = np.std(asr_values)
            stability_scores[method_name] = {'variance': variance, 'std': std}
        
        # 排序（方差越小越稳定）
        sorted_methods = sorted(stability_scores.items(), key=lambda x: x[1]['variance'])
        
        for rank, (method_name, scores) in enumerate(sorted_methods, 1):
            report += f"{rank}. **{method_name}** - Std: {scores['std']:.2f}%, Variance: {scores['variance']:.2f}\n"
        
        report += "\n---\n\n## Recommendations\n\n"
        
        most_stable = sorted_methods[0][0]
        report += f"1. **Most Stable Method:** {most_stable} shows the lowest sensitivity to parameter changes.\n"
        report += f"2. **Optimal Parameters:** Use the parameters identified above for best ASR.\n"
        report += f"3. **Trade-offs:** Consider L0 norm and time when choosing parameters.\n"
        
        report += "\n---\n\n## Visualizations\n\n"
        report += "- `asr_sensitivity_curves.pdf` - ASR vs parameter value\n"
        report += "- `l0_sensitivity_curves.pdf` - L0 norm vs parameter value\n"
        report += "- `time_sensitivity_curves.pdf` - Time vs parameter value\n"
        report += "- `comprehensive_heatmap.pdf` - All metrics heatmap\n"
        
        report += f"\n---\n\n*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        # 保存报告
        report_file = self.output_dir / 'sensitivity_analysis_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ Analysis report saved: {report_file}")
        
        # 也打印摘要到控制台
        print(f"\n{'='*60}")
        print("📊 ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print("\n🏆 Most Stable Methods (by parameter sensitivity):")
        for rank, (method_name, scores) in enumerate(sorted_methods[:3], 1):
            print(f"  {rank}. {method_name} (Std: {scores['std']:.2f}%)")
        
        print("\n🎯 Optimal Parameters:")
        for method_name, method_results in self.results.items():
            best = max(method_results, key=lambda x: x['summary']['asr'])
            param_name = PARAMETER_CONFIGS[method_name]['param_name']
            print(f"  • {method_name}: {param_name}={best['summary']['param_value']} "
                  f"(ASR={best['summary']['asr']:.1f}%)")

def main():
    # 设置随机种子
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    
    # 创建分析器
    analyzer = ParameterSensitivityAnalyzer()
    
    # 运行分析
    analyzer.run_sensitivity_analysis()
    
    print("\n✅ Parameter sensitivity analysis complete!")
    print(f"📁 Results saved to: {CONFIG['output_dir']}")
    
    return 0

if __name__ == '__main__':
    exit(main())

