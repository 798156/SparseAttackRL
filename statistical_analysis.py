"""
统计分析脚本 - Day 6
对所有实验结果进行深入的统计分析
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple

def load_results() -> Dict:
    """加载所有实验结果"""
    results = {
        'ResNet18': json.load(open('results/week1_day1/resnet18_summary.json')),
        'VGG16': json.load(open('results/week1_day2/vgg16_summary.json')),
        'MobileNetV2': json.load(open('results/week1_day5/mobilenetv2_summary.json'))
    }
    return results

def load_detailed_results() -> Dict:
    """加载详细的单样本结果"""
    detailed = {}
    
    # 检查并加载所有可能的详细数据文件
    detailed_files = {
        'ResNet18': 'results/week1_day1/resnet18_detailed.json',
        'VGG16': 'results/week1_day2/vgg16_detailed.json',
        'MobileNetV2': 'results/week1_day5/mobilenetv2_detailed.json'
    }
    
    for model, filepath in detailed_files.items():
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                detailed[model] = json.load(f)
            print(f"  ✅ {model}: 详细数据加载成功")
        else:
            print(f"  ⚠️  {model}: 详细数据不存在，跳过")
    
    return detailed

def extract_attack_results(detailed_data: Dict, attack_name: str) -> List[bool]:
    """提取某个攻击方法的成功/失败列表"""
    results = []
    for sample in detailed_data:
        if attack_name in sample:
            results.append(sample[attack_name]['success'])
    return results

def significance_test_between_methods(detailed: Dict, model: str) -> pd.DataFrame:
    """对同一模型的不同攻击方法进行显著性检验"""
    print(f"\n{'='*80}")
    print(f"📊 {model} - 攻击方法之间的显著性检验")
    print(f"{'='*80}\n")
    
    methods = ['JSMA', 'One-Pixel', 'SparseFool']
    
    # 提取成功率数据
    method_success = {}
    for method in methods:
        success_list = extract_attack_results(detailed[model], method)
        method_success[method] = success_list
        print(f"{method}: {len(success_list)}个样本, ASR={sum(success_list)/len(success_list)*100:.1f}%")
    
    # 两两比较 (McNemar's test - 配对样本)
    results = []
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            method1, method2 = methods[i], methods[j]
            success1 = method_success[method1]
            success2 = method_success[method2]
            
            # McNemar's test (适用于配对二分类数据)
            # 构建混淆矩阵
            both_success = sum(s1 and s2 for s1, s2 in zip(success1, success2))
            both_fail = sum(not s1 and not s2 for s1, s2 in zip(success1, success2))
            only_1 = sum(s1 and not s2 for s1, s2 in zip(success1, success2))
            only_2 = sum(not s1 and s2 for s1, s2 in zip(success1, success2))
            
            # McNemar统计量
            if only_1 + only_2 > 0:
                statistic = (abs(only_1 - only_2) - 1) ** 2 / (only_1 + only_2)
                p_value = 1 - stats.chi2.cdf(statistic, 1)
            else:
                statistic = 0
                p_value = 1.0
            
            significance = '✅ 显著' if p_value < 0.05 else '❌ 不显著'
            
            results.append({
                '对比': f'{method1} vs {method2}',
                '统计量': f'{statistic:.3f}',
                'p值': f'{p_value:.4f}',
                '结论': significance
            })
            
            print(f"\n{method1} vs {method2}:")
            print(f"  两者都成功: {both_success}")
            print(f"  两者都失败: {both_fail}")
            print(f"  仅{method1}成功: {only_1}")
            print(f"  仅{method2}成功: {only_2}")
            print(f"  McNemar统计量: {statistic:.3f}")
            print(f"  p值: {p_value:.4f} {significance}")
    
    return pd.DataFrame(results)

def significance_test_between_models(detailed: Dict, method: str) -> pd.DataFrame:
    """对同一攻击方法在不同模型上的表现进行显著性检验"""
    print(f"\n{'='*80}")
    print(f"📊 {method} - 不同模型之间的显著性检验")
    print(f"{'='*80}\n")
    
    models = ['ResNet18', 'VGG16', 'MobileNetV2']
    
    # 提取成功率数据
    model_success = {}
    for model in models:
        if model in detailed:
            success_list = extract_attack_results(detailed[model], method)
            model_success[model] = success_list
            print(f"{model}: {len(success_list)}个样本, ASR={sum(success_list)/len(success_list)*100:.1f}%")
    
    # 两两比较
    results = []
    model_list = list(model_success.keys())
    for i in range(len(model_list)):
        for j in range(i+1, len(model_list)):
            model1, model2 = model_list[i], model_list[j]
            success1 = model_success[model1]
            success2 = model_success[model2]
            
            # 确保样本数相同
            min_len = min(len(success1), len(success2))
            success1 = success1[:min_len]
            success2 = success2[:min_len]
            
            # McNemar's test
            both_success = sum(s1 and s2 for s1, s2 in zip(success1, success2))
            both_fail = sum(not s1 and not s2 for s1, s2 in zip(success1, success2))
            only_1 = sum(s1 and not s2 for s1, s2 in zip(success1, success2))
            only_2 = sum(not s1 and s2 for s1, s2 in zip(success1, success2))
            
            if only_1 + only_2 > 0:
                statistic = (abs(only_1 - only_2) - 1) ** 2 / (only_1 + only_2)
                p_value = 1 - stats.chi2.cdf(statistic, 1)
            else:
                statistic = 0
                p_value = 1.0
            
            significance = '✅ 显著' if p_value < 0.05 else '❌ 不显著'
            
            results.append({
                '对比': f'{model1} vs {model2}',
                '统计量': f'{statistic:.3f}',
                'p值': f'{p_value:.4f}',
                '结论': significance
            })
            
            print(f"\n{model1} vs {model2}:")
            print(f"  两者都成功: {both_success}")
            print(f"  两者都失败: {both_fail}")
            print(f"  仅{model1}成功: {only_1}")
            print(f"  仅{model2}成功: {only_2}")
            print(f"  McNemar统计量: {statistic:.3f}")
            print(f"  p值: {p_value:.4f} {significance}")
    
    return pd.DataFrame(results)

def analyze_failure_cases(detailed: Dict) -> Dict:
    """分析失败案例"""
    print(f"\n{'='*80}")
    print(f"🔍 失败案例分析")
    print(f"{'='*80}\n")
    
    failure_stats = {}
    
    for model in detailed.keys():
        print(f"\n【{model}】")
        model_data = detailed[model]
        
        # 统计每个样本的失败情况
        all_fail = 0  # 所有攻击都失败
        partial_fail = 0  # 部分攻击失败
        all_success = 0  # 所有攻击都成功
        
        all_fail_samples = []
        
        for sample in model_data:
            methods = ['JSMA', 'One-Pixel', 'SparseFool']
            success_count = sum(1 for m in methods if m in sample and sample[m]['success'])
            
            if success_count == 0:
                all_fail += 1
                all_fail_samples.append(sample['sample_id'])
            elif success_count == len(methods):
                all_success += 1
            else:
                partial_fail += 1
        
        total = len(model_data)
        print(f"  总样本数: {total}")
        print(f"  所有攻击都成功: {all_success} ({all_success/total*100:.1f}%)")
        print(f"  部分攻击失败: {partial_fail} ({partial_fail/total*100:.1f}%)")
        print(f"  所有攻击都失败: {all_fail} ({all_fail/total*100:.1f}%)")
        
        if all_fail > 0:
            print(f"  完全失败的样本ID: {all_fail_samples[:10]}{'...' if len(all_fail_samples) > 10 else ''}")
        
        failure_stats[model] = {
            'all_success': all_success,
            'partial_fail': partial_fail,
            'all_fail': all_fail,
            'all_fail_samples': all_fail_samples
        }
    
    return failure_stats

def analyze_l0_distribution(detailed: Dict) -> None:
    """分析L0分布"""
    print(f"\n{'='*80}")
    print(f"📏 L0分布分析")
    print(f"{'='*80}\n")
    
    for model in detailed.keys():
        print(f"\n【{model}】")
        model_data = detailed[model]
        
        for method in ['JSMA', 'One-Pixel', 'SparseFool']:
            l0_values = []
            for sample in model_data:
                if method in sample and sample[method]['success']:
                    l0_values.append(sample[method]['l0'])
            
            if len(l0_values) > 0:
                print(f"\n  {method}:")
                print(f"    成功攻击数: {len(l0_values)}")
                print(f"    L0均值: {np.mean(l0_values):.2f}")
                print(f"    L0标准差: {np.std(l0_values):.2f}")
                print(f"    L0中位数: {np.median(l0_values):.2f}")
                print(f"    L0范围: [{np.min(l0_values):.0f}, {np.max(l0_values):.0f}]")
                print(f"    L0分布: 25%={np.percentile(l0_values, 25):.1f}, "
                      f"75%={np.percentile(l0_values, 75):.1f}")

def correlation_analysis(results: Dict) -> None:
    """相关性分析"""
    print(f"\n{'='*80}")
    print(f"📈 相关性分析")
    print(f"{'='*80}\n")
    
    # 准确率 vs 平均ASR
    models = ['ResNet18', 'VGG16', 'MobileNetV2']
    accuracies = [83.77, 92.27, 84.90]
    
    # 计算每个模型的平均ASR
    avg_asrs = []
    for model in models:
        model_results = results[model]
        asrs = []
        for method in ['JSMA', 'One-Pixel', 'SparseFool']:
            if method in model_results and model_results[method]['ASR'] > 0:
                asrs.append(model_results[method]['ASR'])
        avg_asrs.append(np.mean(asrs) if asrs else 0)
    
    # 计算相关系数
    corr, p_value = stats.pearsonr(accuracies, avg_asrs)
    
    print("模型准确率 vs 平均ASR:")
    for model, acc, asr in zip(models, accuracies, avg_asrs):
        print(f"  {model}: 准确率={acc:.2f}%, 平均ASR={asr:.1f}%")
    
    print(f"\n  Pearson相关系数: {corr:.3f}")
    print(f"  p值: {p_value:.4f}")
    
    if corr < 0 and p_value < 0.1:
        print(f"  ✅ 存在负相关：准确率越高，ASR越低（鲁棒性越强）")
    else:
        print(f"  ⚠️  相关性不显著")

def main():
    """主函数"""
    print("\n" + "="*80)
    print("🔬 统计分析报告")
    print("="*80)
    
    # 加载数据
    print("\n📦 加载数据...")
    results = load_results()
    detailed = load_detailed_results()
    print(f"✅ 加载完成：{len(results)}个模型的数据")
    
    # 1. 相关性分析
    correlation_analysis(results)
    
    # 2. 不同攻击方法之间的显著性检验
    for model in ['ResNet18', 'VGG16', 'MobileNetV2']:
        if model in detailed:
            df = significance_test_between_methods(detailed, model)
            print(f"\n{model} 显著性检验汇总:")
            print(df.to_string(index=False))
        else:
            print(f"\n⚠️  {model}: 无详细数据，跳过方法间检验")
    
    # 3. 不同模型之间的显著性检验
    for method in ['JSMA', 'SparseFool']:  # One-Pixel数据不完整，暂时跳过
        df = significance_test_between_models(detailed, method)
        print(f"\n{method} 显著性检验汇总:")
        print(df.to_string(index=False))
    
    # 4. 失败案例分析
    failure_stats = analyze_failure_cases(detailed)
    
    # 5. L0分布分析
    analyze_l0_distribution(detailed)
    
    # 保存分析报告
    print(f"\n{'='*80}")
    print("💾 保存分析报告...")
    
    output_dir = Path('results/statistical_analysis')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 保存失败案例统计
    with open(output_dir / 'failure_cases.json', 'w') as f:
        json.dump(failure_stats, f, indent=2)
    
    print(f"✅ 报告已保存到: {output_dir}")
    
    print(f"\n{'='*80}")
    print("🎉 统计分析完成！")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

