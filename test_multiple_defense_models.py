#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试多个防御模型
对比不同防御策略对L0攻击的鲁棒性
"""

import torch
import torchvision
import torchvision.transforms as transforms
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm

try:
    from robustbench import load_model
    ROBUSTBENCH_AVAILABLE = True
except ImportError:
    ROBUSTBENCH_AVAILABLE = False
    print("⚠️ RobustBench未安装，无法继续")

from attack_adapters import (
    jsma_attack_adapter,
    sparsefool_attack_adapter,
    greedy_attack_adapter,
    pixel_gradient_attack_adapter,
    random_sparse_attack_adapter
)

class MultiDefenseModelTester:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path('results/multi_defense_models')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 攻击方法配置
        self.attack_methods = {
            'jsma': {
                'func': jsma_attack_adapter,
                'config': {'max_pixels': 10, 'theta': 1.0, 'max_iterations': 100}
            },
            'sparsefool': {
                'func': sparsefool_attack_adapter,
                'config': {'max_iter': 20, 'overshoot': 0.02, 'lambda_': 3.0}
            },
            'greedy': {
                'func': greedy_attack_adapter,
                'config': {'max_pixels': 10, 'alpha': 0.1, 'max_iterations': 100}
            },
            'pixelgrad': {
                'func': pixel_gradient_attack_adapter,
                'config': {'max_pixels': 10, 'alpha': 0.2, 'beta': 0.9}
            },
            'randomsparse': {
                'func': random_sparse_attack_adapter,
                'config': {'max_pixels': 10, 'perturbation_size': 0.2, 'max_attempts': 50}
            }
        }
        
        # 防御模型列表（重新测试Rice2020确保公平对比）
        self.defense_models = {
            'Rice2020Overfitting': {
                'model_name': 'Rice2020Overfitting',
                'description': 'TRADES强防御（强L∞鲁棒性）',
                'threat_model': 'Linf'
            }
        }
        
        print(f"Device: {self.device}")
        print(f"测试 {len(self.defense_models)} 个防御模型")
        print(f"使用 {len(self.attack_methods)} 种攻击方法")
    
    def load_test_data(self, num_samples=100):
        """加载测试数据"""
        print("\n" + "="*60)
        print(f"📂 加载CIFAR-10测试集（目标：{num_samples}个样本）...")
        print("="*60)
        
        # 设置随机种子
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
        
        return test_loader
    
    def select_samples(self, model, test_loader, num_samples=100):
        """选择正确分类的样本"""
        print(f"\n🎯 选择{num_samples}个被模型正确分类的样本...")
        
        samples = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                if len(samples) >= num_samples:
                    break
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                pred = outputs.argmax(dim=1)
                
                if pred.item() == labels.item():
                    samples.append({
                        'image': images.cpu(),
                        'label': labels.item()
                    })
        
        print(f"✓ 选择完成！共{len(samples)}个样本")
        return samples
    
    def test_attack_on_samples(self, model, samples, attack_name, attack_func, attack_config):
        """在样本上测试攻击"""
        results = []
        success_count = 0
        
        l0_values = []
        l2_values = []
        ssim_values = []
        time_values = []
        
        for i, sample in enumerate(tqdm(samples, desc=f"  {attack_name}")):
            image = sample['image'].to(self.device)
            label = sample['label']
            label_tensor = torch.tensor([label]).to(self.device)
            
            try:
                import time
                start_time = time.time()
                
                adv_image, success, info = attack_func(
                    model, image, label_tensor, device=self.device, **attack_config
                )
                
                elapsed = time.time() - start_time
                
                result = {
                    'sample_id': i,
                    'success': bool(success),
                    'time': float(elapsed)
                }
                
                if success:
                    result['l0'] = float(info.get('l0', 0))
                    result['l2'] = float(info.get('l2', 0))
                    result['ssim'] = float(info.get('ssim', 0))
                    
                    l0_values.append(result['l0'])
                    l2_values.append(result['l2'])
                    ssim_values.append(result['ssim'])
                    time_values.append(result['time'])
                    
                    success_count += 1
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    'sample_id': i,
                    'success': False,
                    'error': str(e)[:100]
                })
        
        # 计算统计数据
        summary = {
            'method': attack_name,
            'asr': (success_count / len(samples)) * 100 if samples else 0,
            'success_count': success_count,
            'total_samples': len(samples),
            'avg_l0': float(np.mean(l0_values)) if l0_values else 0,
            'avg_l2': float(np.mean(l2_values)) if l2_values else 0,
            'avg_ssim': float(np.mean(ssim_values)) if ssim_values else 0,
            'avg_time': float(np.mean(time_values)) if time_values else 0,
            'std_l0': float(np.std(l0_values)) if l0_values else 0,
            'std_l2': float(np.std(l2_values)) if l2_values else 0
        }
        
        return results, summary
    
    def test_defense_model(self, model_key, num_samples=100):
        """测试单个防御模型"""
        print("\n" + "🚀"*30)
        print(f"测试防御模型: {model_key}")
        print(f"描述: {self.defense_models[model_key]['description']}")
        print("🚀"*30)
        
        # 加载模型
        model_name = self.defense_models[model_key]['model_name']
        threat_model = self.defense_models[model_key]['threat_model']
        
        try:
            print(f"\n📥 加载模型: {model_name}...")
            model = load_model(
                model_name=model_name,
                dataset='cifar10',
                threat_model=threat_model
            )
            model = model.to(self.device)
            model.eval()
            print("✓ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return None
        
        # 加载测试数据
        test_loader = self.load_test_data(num_samples)
        
        # 选择样本
        samples = self.select_samples(model, test_loader, num_samples)
        
        if len(samples) < num_samples:
            print(f"⚠️  警告：只找到{len(samples)}个正确分类的样本")
        
        # 测试所有攻击方法
        all_results = {}
        all_summaries = {}
        
        print("\n" + "="*60)
        print("📊 开始攻击测试...")
        print("="*60)
        
        for attack_name, attack_info in self.attack_methods.items():
            results, summary = self.test_attack_on_samples(
                model, samples, attack_name,
                attack_info['func'], attack_info['config']
            )
            all_results[attack_name] = results
            all_summaries[attack_name] = summary
            
            print(f"  ✓ {attack_name.upper()}: ASR={summary['asr']:.1f}%")
        
        # 保存结果
        output_data = {
            'defense_model': model_key,
            'model_name': model_name,
            'description': self.defense_models[model_key]['description'],
            'test_samples': len(samples),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summaries': all_summaries,
            'detailed_results': all_results
        }
        
        output_file = self.output_dir / f'{model_key.lower()}_results.json'
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ 结果已保存: {output_file}")
        
        return all_summaries
    
    def run_all_tests(self, num_samples=100):
        """运行所有防御模型测试"""
        print("\n" + "🎯"*30)
        print(f"开始测试 {len(self.defense_models)} 个防御模型")
        print("🎯"*30)
        
        all_model_results = {}
        
        for model_key in self.defense_models.keys():
            try:
                summaries = self.test_defense_model(model_key, num_samples)
                if summaries:
                    all_model_results[model_key] = summaries
            except Exception as e:
                print(f"\n❌ 测试 {model_key} 时出错: {e}")
                continue
        
        # 生成对比报告
        self.generate_comparison_report(all_model_results)
        
        print("\n" + "🎉"*30)
        print("所有测试完成！")
        print("🎉"*30)
        print(f"\n📁 结果保存在: {self.output_dir}")
    
    def generate_comparison_report(self, all_results):
        """生成对比报告"""
        print("\n" + "="*60)
        print("📊 生成对比报告...")
        print("="*60)
        
        report = f"""# 多防御模型对比报告

**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**测试的防御模型数量:** {len(all_results)}  
**攻击方法数量:** 5

---

## 1. 测试的防御模型

"""
        
        for i, (model_key, model_info) in enumerate(self.defense_models.items(), 1):
            if model_key in all_results:
                report += f"{i}. **{model_key}:** {model_info['description']}\n"
        
        report += "\n---\n\n## 2. 攻击成功率（ASR）对比\n\n"
        report += "| 防御模型 | JSMA | SparseFool | Greedy | PixelGrad | RandomSparse |\n"
        report += "|----------|------|------------|--------|-----------|---------------|\n"
        
        for model_key, summaries in all_results.items():
            row = [model_key]
            for method in ['jsma', 'sparsefool', 'greedy', 'pixelgrad', 'randomsparse']:
                asr = summaries.get(method, {}).get('asr', 0)
                row.append(f"{asr:.1f}%")
            report += "| " + " | ".join(row) + " |\n"
        
        report += "\n---\n\n## 3. 平均L0范数对比\n\n"
        report += "| 防御模型 | JSMA | SparseFool | Greedy | PixelGrad | RandomSparse |\n"
        report += "|----------|------|------------|--------|-----------|---------------|\n"
        
        for model_key, summaries in all_results.items():
            row = [model_key]
            for method in ['jsma', 'sparsefool', 'greedy', 'pixelgrad', 'randomsparse']:
                l0 = summaries.get(method, {}).get('avg_l0', 0)
                row.append(f"{l0:.2f}")
            report += "| " + " | ".join(row) + " |\n"
        
        report += "\n---\n\n## 4. 关键发现\n\n"
        
        # 找出最鲁棒的模型
        avg_asrs = {}
        for model_key, summaries in all_results.items():
            asrs = [s.get('asr', 0) for s in summaries.values()]
            avg_asrs[model_key] = np.mean(asrs) if asrs else 0
        
        most_robust = min(avg_asrs.items(), key=lambda x: x[1])
        least_robust = max(avg_asrs.items(), key=lambda x: x[1])
        
        report += f"### 4.1 防御模型鲁棒性排名\n\n"
        report += "按平均ASR排序（越低越鲁棒）：\n\n"
        
        for rank, (model_key, avg_asr) in enumerate(sorted(avg_asrs.items(), key=lambda x: x[1]), 1):
            report += f"{rank}. **{model_key}:** {avg_asr:.1f}% 平均ASR\n"
        
        report += f"\n### 4.2 主要洞察\n\n"
        report += f"1. **最鲁棒模型:** {most_robust[0]} (平均ASR: {most_robust[1]:.1f}%)\n"
        report += f"2. **最脆弱模型:** {least_robust[0]} (平均ASR: {least_robust[1]:.1f}%)\n"
        report += f"3. **鲁棒性差距:** {least_robust[1] - most_robust[1]:.1f} 个百分点\n\n"
        
        report += "### 4.3 方法-模型交互\n\n"
        report += "分析不同防御策略对不同攻击方法的效果差异...\n\n"
        
        report += f"\n---\n\n*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        # 保存报告
        report_file = self.output_dir / 'multi_defense_comparison.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ 对比报告已保存: {report_file}")

def main():
    if not ROBUSTBENCH_AVAILABLE:
        print("❌ 请先安装RobustBench:")
        print("   pip install git+https://github.com/RobustBench/robustbench.git")
        return 1
    
    tester = MultiDefenseModelTester()
    
    # 运行测试（默认100个样本）
    tester.run_all_tests(num_samples=100)
    
    return 0

if __name__ == '__main__':
    exit(main())

