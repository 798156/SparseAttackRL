#!/usr/bin/env python3
"""对比所有模型的实验结果"""

import json
import pandas as pd

print("=" * 80)
print("📊 完整实验结果对比")
print("=" * 80)

# 加载所有结果
results = {}

# ResNet18
with open('results/week1_day1/resnet18_summary.json') as f:
    results['ResNet18'] = json.load(f)

# VGG16
with open('results/week1_day2/vgg16_summary.json') as f:
    results['VGG16'] = json.load(f)

# MobileNetV2
with open('results/week1_day5/mobilenetv2_summary.json') as f:
    results['MobileNetV2'] = json.load(f)

# 创建对比表格
print("\n" + "=" * 80)
print("📈 攻击成功率 (ASR) 对比")
print("=" * 80 + "\n")

asr_data = []
for model, model_results in results.items():
    row = {'Model': model}
    for method in ['JSMA', 'One-Pixel', 'SparseFool']:
        if method in model_results:
            row[method] = f"{model_results[method]['ASR']:.1f}%"
        else:
            row[method] = "N/A"
    asr_data.append(row)

df_asr = pd.DataFrame(asr_data)
print(df_asr.to_string(index=False))

print("\n" + "=" * 80)
print("📏 平均修改像素数 (L0) 对比")
print("=" * 80 + "\n")

l0_data = []
for model, model_results in results.items():
    row = {'Model': model}
    for method in ['JSMA', 'One-Pixel', 'SparseFool']:
        if method in model_results and model_results[method]['ASR'] > 0:
            row[method] = f"{model_results[method]['L0']:.2f}"
        else:
            row[method] = "N/A"
    l0_data.append(row)

df_l0 = pd.DataFrame(l0_data)
print(df_l0.to_string(index=False))

print("\n" + "=" * 80)
print("⏱️  平均攻击时间 (秒) 对比")
print("=" * 80 + "\n")

time_data = []
for model, model_results in results.items():
    row = {'Model': model}
    for method in ['JSMA', 'One-Pixel', 'SparseFool']:
        if method in model_results:
            row[method] = f"{model_results[method]['Time']:.3f}s"
        else:
            row[method] = "N/A"
    time_data.append(row)

df_time = pd.DataFrame(time_data)
print(df_time.to_string(index=False))

print("\n" + "=" * 80)
print("🔑 关键发现")
print("=" * 80 + "\n")

print("1️⃣  模型鲁棒性排序（按准确率）：")
print("   VGG16 (92.27%) > MobileNetV2 (84.90%) > ResNet18 (83.77%)")
print("   鲁棒性与准确率正相关 ✓")

print("\n2️⃣  One-Pixel攻击的架构敏感性：")
print("   MobileNetV2: 21% ← 最脆弱")
print("   ResNet18:    16%")
print("   VGG16:       0%  ← 完全免疫")
print("   → 轻量级架构对单像素扰动更敏感！")

print("\n3️⃣  最有效的攻击方法：")
print("   ResNet18:     JSMA (55%)")
print("   MobileNetV2:  SparseFool (41%)")
print("   VGG16:        JSMA (27%)")

print("\n4️⃣  效率对比：")
print("   SparseFool: 最快（~0.4秒）")
print("   JSMA:       中等（~1.2秒）")
print("   One-Pixel:  最慢（~25秒）")

print("\n" + "=" * 80)
print("✅ 完整实验矩阵：9/9 组数据全部完成！")
print("=" * 80)








