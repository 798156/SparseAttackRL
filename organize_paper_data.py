#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文数据整理脚本
从实验结果中提取数据，格式化为论文表格
"""

import json
from pathlib import Path
import numpy as np

class PaperDataOrganizer:
    def __init__(self):
        self.results_dir = Path('results')
        self.output_dir = Path('paper_data_summary')
        self.output_dir.mkdir(exist_ok=True)
        
    def load_standard_model_results(self):
        """加载标准模型结果（Week 1）"""
        baseline_dir = self.results_dir / 'complete_baseline'
        
        models = ['resnet18', 'vgg16', 'mobilenetv2']
        methods = ['jsma', 'sparsefool', 'greedy', 'pixelgrad', 'randomsparse']
        
        results = {}
        
        for model in models:
            results[model] = {}
            for method in methods:
                json_file = baseline_dir / f'{model}_{method}.json'
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        results[model][method] = self._extract_summary(data)
                else:
                    print(f"⚠️  文件不存在: {json_file}")
        
        return results
    
    def load_defended_model_results(self):
        """加载防御模型结果（Week 1）"""
        defended_dir = self.results_dir / 'defended_model'
        
        methods = ['jsma', 'sparsefool', 'greedy', 'pixelgrad', 'randomsparse']
        
        results = {}
        
        for method in methods:
            json_file = defended_dir / f'defended_{method}.json'
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    results[method] = self._extract_summary(data)
            else:
                print(f"⚠️  文件不存在: {json_file}")
        
        return results
    
    def load_sensitivity_results(self):
        """加载参数敏感性结果"""
        sensitivity_dir = self.results_dir / 'parameter_sensitivity'
        
        methods = ['jsma', 'sparsefool', 'greedy', 'pixelgrad', 'randomsparse']
        
        results = {}
        
        for method in methods:
            json_file = sensitivity_dir / f'{method}_sensitivity.json'
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    results[method] = data
            else:
                print(f"⚠️  文件不存在: {json_file}")
        
        return results
    
    def _extract_summary(self, data):
        """从JSON数据中提取摘要信息"""
        # Week 1格式：直接在顶层有asr, avg_l0等字段
        if 'asr' in data:
            return {
                'asr': round(float(data['asr']), 1),
                'avg_l0': round(float(data.get('avg_l0', 0)), 2),
                'avg_l2': round(float(data.get('avg_l2', 0)), 4),
                'avg_time': round(float(data.get('avg_time', 0)), 3),
                'num_samples': int(data.get('total_samples', 0)),
                'num_success': int(data.get('success_count', 0))
            }
        
        # 防御模型格式：可能有samples数组
        if 'samples' in data:
            samples = data['samples']
            successful = [s for s in samples if s.get('success', False)]
            
            asr = (len(successful) / len(samples) * 100) if samples else 0
            
            if successful:
                l0_values = [s.get('l0_norm', 0) for s in successful if s.get('l0_norm', 0) > 0]
                l2_values = [s.get('l2_norm', 0) for s in successful if s.get('l2_norm', 0) > 0]
                time_values = [s.get('time', 0) for s in successful]
                
                avg_l0 = np.mean(l0_values) if l0_values else 0
                avg_l2 = np.mean(l2_values) if l2_values else 0
                avg_time = np.mean(time_values) if time_values else 0
            else:
                avg_l0 = avg_l2 = avg_time = 0
            
            return {
                'asr': round(asr, 1),
                'avg_l0': round(avg_l0, 2),
                'avg_l2': round(avg_l2, 4),
                'avg_time': round(avg_time, 3),
                'num_samples': len(samples),
                'num_success': len(successful)
            }
        
        return None
    
    def generate_section_6_1_table(self, results):
        """生成6.1节表格：标准模型结果"""
        print("\n" + "="*60)
        print("📊 Section 6.1: 标准模型攻击效果")
        print("="*60)
        
        methods = ['JSMA', 'SparseFool', 'Greedy', 'PixelGrad', 'RandomSparse']
        method_keys = ['jsma', 'sparsefool', 'greedy', 'pixelgrad', 'randomsparse']
        models = ['ResNet18', 'VGG16', 'MobileNetV2']
        model_keys = ['resnet18', 'vgg16', 'mobilenetv2']
        
        output = []
        
        # 为每个模型生成表格
        for model_name, model_key in zip(models, model_keys):
            output.append(f"\n### **{model_name} 结果**\n")
            output.append("| 方法 | ASR (%) | 平均L0 | 平均L2 | 平均时间(s) | 效率比(ASR/Time) |")
            output.append("|------|---------|--------|--------|-------------|------------------|")
            
            model_results = results.get(model_key, {})
            
            for method_name, method_key in zip(methods, method_keys):
                method_result = model_results.get(method_key)
                if method_result:
                    asr = method_result['asr']
                    l0 = method_result['avg_l0']
                    l2 = method_result['avg_l2']
                    time = method_result['avg_time']
                    efficiency = round(asr / time, 1) if time > 0 else 0
                    
                    output.append(f"| {method_name} | {asr:.1f} | {l0:.2f} | {l2:.4f} | {time:.3f} | {efficiency:.1f} |")
                else:
                    output.append(f"| {method_name} | - | - | - | - | - |")
        
        # 交叉模型平均
        output.append(f"\n### **跨模型平均**\n")
        output.append("| 方法 | 平均ASR (%) | 平均L0 | 平均L2 | 平均时间(s) |")
        output.append("|------|-------------|--------|--------|-------------|")
        
        for method_name, method_key in zip(methods, method_keys):
            asr_values = []
            l0_values = []
            l2_values = []
            time_values = []
            
            for model_key in model_keys:
                method_result = results.get(model_key, {}).get(method_key)
                if method_result and method_result['asr'] > 0:
                    asr_values.append(method_result['asr'])
                    l0_values.append(method_result['avg_l0'])
                    l2_values.append(method_result['avg_l2'])
                    time_values.append(method_result['avg_time'])
            
            if asr_values:
                avg_asr = np.mean(asr_values)
                avg_l0 = np.mean(l0_values)
                avg_l2 = np.mean(l2_values)
                avg_time = np.mean(time_values)
                
                output.append(f"| {method_name} | {avg_asr:.1f} | {avg_l0:.2f} | {avg_l2:.4f} | {avg_time:.3f} |")
            else:
                output.append(f"| {method_name} | - | - | - | - |")
        
        result_text = '\n'.join(output)
        print(result_text)
        
        # 保存
        output_file = self.output_dir / 'section_6_1_tables.md'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result_text)
        print(f"\n✅ 已保存到: {output_file}")
        
        return result_text
    
    def generate_section_6_2_summary(self, sensitivity_results):
        """生成6.2节摘要：参数敏感性"""
        print("\n" + "="*60)
        print("📊 Section 6.2: 参数敏感性分析摘要")
        print("="*60)
        
        output = []
        
        output.append("\n### **最优参数配置**\n")
        output.append("| 方法 | 最优参数 | 最优值 | 达到的ASR | 平均L0 | 平均时间 |")
        output.append("|------|----------|--------|-----------|--------|----------|")
        
        method_display = {
            'jsma': 'JSMA',
            'sparsefool': 'SparseFool',
            'greedy': 'Greedy',
            'pixelgrad': 'PixelGrad',
            'randomsparse': 'RandomSparse'
        }
        
        for method_key, method_name in method_display.items():
            if method_key in sensitivity_results:
                configs = sensitivity_results[method_key]
                # 找最高ASR的配置
                best_config = max(configs, key=lambda x: x['summary']['asr'])
                summary = best_config['summary']
                
                param_name = summary['param_name']
                param_value = summary['param_value']
                asr = summary['asr']
                l0 = summary['avg_l0']
                time = summary['avg_time']
                
                output.append(f"| {method_name} | {param_name}={param_value} | {param_value} | {asr:.1f}% | {l0:.2f} | {time:.3f}s |")
        
        # 稳定性排名
        output.append("\n### **稳定性排名（标准差从小到大）**\n")
        output.append("| 排名 | 方法 | 标准差 | 方差 | 解释 |")
        output.append("|------|------|--------|------|------|")
        
        stability_scores = {}
        for method_key in sensitivity_results:
            configs = sensitivity_results[method_key]
            asr_values = [c['summary']['asr'] for c in configs]
            std = np.std(asr_values)
            var = np.var(asr_values)
            stability_scores[method_key] = {'std': std, 'var': var}
        
        # 排序
        sorted_methods = sorted(stability_scores.items(), key=lambda x: x[1]['std'])
        
        stability_desc = {
            1: "非常稳定",
            2: "稳定",
            3: "中等",
            4: "较敏感",
            5: "高度敏感"
        }
        
        for rank, (method_key, scores) in enumerate(sorted_methods, 1):
            method_name = method_display[method_key]
            desc = stability_desc.get(rank, "敏感")
            output.append(f"| {rank} | {method_name} | {scores['std']:.2f}% | {scores['var']:.2f} | {desc} |")
        
        result_text = '\n'.join(output)
        print(result_text)
        
        # 保存
        output_file = self.output_dir / 'section_6_2_summary.md'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result_text)
        print(f"\n✅ 已保存到: {output_file}")
        
        return result_text
    
    def generate_section_6_3_comparison(self, standard_results, defended_results):
        """生成6.3节对比：标准vs防御模型"""
        print("\n" + "="*60)
        print("📊 Section 6.3: 标准模型 vs 防御模型对比")
        print("="*60)
        
        output = []
        
        # 使用ResNet18作为标准模型基准
        standard_model = 'resnet18'
        
        output.append("\n### **ASR对比（ResNet18标准 vs ResNet18防御）**\n")
        output.append("| 方法 | 标准模型ASR | 防御模型ASR | 下降 | 下降率 | 鲁棒性排名 |")
        output.append("|------|-------------|-------------|------|--------|------------|")
        
        methods = ['JSMA', 'SparseFool', 'Greedy', 'PixelGrad', 'RandomSparse']
        method_keys = ['jsma', 'sparsefool', 'greedy', 'pixelgrad', 'randomsparse']
        
        drop_data = []
        
        for method_name, method_key in zip(methods, method_keys):
            standard = standard_results.get(standard_model, {}).get(method_key)
            defended = defended_results.get(method_key)
            
            if standard and defended:
                std_asr = standard['asr']
                def_asr = defended['asr']
                drop = std_asr - def_asr
                drop_rate = (drop / std_asr * 100) if std_asr > 0 else 0
                
                drop_data.append((method_name, std_asr, def_asr, drop, drop_rate))
        
        # 按下降率排序（下降越少=越鲁棒）
        drop_data_sorted = sorted(drop_data, key=lambda x: x[4])
        
        for rank, (method_name, std_asr, def_asr, drop, drop_rate) in enumerate(drop_data_sorted, 1):
            output.append(f"| {method_name} | {std_asr:.1f}% | {def_asr:.1f}% | {drop:.1f}% | {drop_rate:.1f}% | #{rank} |")
        
        # 关键发现
        output.append("\n### **关键发现**\n")
        
        if drop_data_sorted:
            most_robust = drop_data_sorted[0]
            least_robust = drop_data_sorted[-1]
            
            output.append(f"1. **最鲁棒方法**: {most_robust[0]} (ASR仅下降{most_robust[4]:.1f}%)")
            output.append(f"2. **最脆弱方法**: {least_robust[0]} (ASR下降{least_robust[4]:.1f}%)")
            
            avg_drop = np.mean([d[4] for d in drop_data_sorted])
            output.append(f"3. **平均ASR下降**: {avg_drop:.1f}%")
            
            # L0攻击威胁
            avg_defended_asr = np.mean([d[2] for d in drop_data_sorted])
            output.append(f"4. **L0攻击对L∞防御的残余威胁**: 平均{avg_defended_asr:.1f}% ASR")
        
        result_text = '\n'.join(output)
        print(result_text)
        
        # 保存
        output_file = self.output_dir / 'section_6_3_comparison.md'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result_text)
        print(f"\n✅ 已保存到: {output_file}")
        
        return result_text
    
    def generate_complete_summary(self):
        """生成完整摘要报告"""
        print("\n" + "🚀"*30)
        print("开始整理论文数据...")
        print("🚀"*30)
        
        # 加载所有数据
        print("\n📂 加载数据...")
        standard_results = self.load_standard_model_results()
        defended_results = self.load_defended_model_results()
        sensitivity_results = self.load_sensitivity_results()
        
        # 生成各节内容
        section_6_1 = self.generate_section_6_1_table(standard_results)
        section_6_2 = self.generate_section_6_2_summary(sensitivity_results)
        section_6_3 = self.generate_section_6_3_comparison(standard_results, defended_results)
        
        # 生成完整摘要
        print("\n" + "="*60)
        print("📝 生成完整数据摘要")
        print("="*60)
        
        summary = []
        summary.append("# 论文数据完整摘要\n")
        summary.append(f"**生成时间**: {Path.cwd()}\n")
        summary.append("---\n")
        
        summary.append("## 数据来源\n")
        summary.append("- Week 1: 标准模型攻击（1500个样本）")
        summary.append("- Week 1: 防御模型攻击（500个样本）")
        summary.append("- Week 2: 参数敏感性分析（1000个样本）")
        summary.append("- **总计**: 3000个对抗样本测试\n")
        
        summary.append("---\n")
        summary.append("## Section 6.1: 标准模型攻击效果\n")
        summary.append(section_6_1)
        
        summary.append("\n---\n")
        summary.append("## Section 6.2: 参数敏感性分析\n")
        summary.append(section_6_2)
        
        summary.append("\n---\n")
        summary.append("## Section 6.3: 防御模型鲁棒性\n")
        summary.append(section_6_3)
        
        summary.append("\n---\n")
        summary.append("## 使用说明\n")
        summary.append("1. 将对应章节的表格复制到论文草稿中")
        summary.append("2. 根据需要调整格式和文字说明")
        summary.append("3. 补充分析和讨论\n")
        
        summary_text = '\n'.join(summary)
        
        # 保存完整摘要
        output_file = self.output_dir / 'complete_data_summary.md'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        print(f"\n✅ 完整摘要已保存到: {output_file}")
        
        print("\n" + "🎉"*30)
        print("数据整理完成！")
        print("🎉"*30)
        
        print("\n📁 生成的文件：")
        print(f"  1. {self.output_dir / 'section_6_1_tables.md'}")
        print(f"  2. {self.output_dir / 'section_6_2_summary.md'}")
        print(f"  3. {self.output_dir / 'section_6_3_comparison.md'}")
        print(f"  4. {self.output_dir / 'complete_data_summary.md'}")
        
        print("\n📊 下一步：")
        print("  1. 查看生成的markdown文件")
        print("  2. 复制内容到论文草稿对应章节")
        print("  3. 补充分析文字")
        
        return summary_text

def main():
    organizer = PaperDataOrganizer()
    organizer.generate_complete_summary()
    return 0

if __name__ == '__main__':
    exit(main())

