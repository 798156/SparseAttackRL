# 🚀 Day 1-2 快速开始指南

## ✅ 已完成的准备工作

1. ✅ 修改了 `final_test_with_greedy.py` - 样本数改为100
2. ✅ 修改了 `test_new_2methods.py` - 样本数改为100
3. ✅ 创建了自动化测试脚本 `run_100_samples_test.py`

---

## 🎯 现在可以开始测试了！

### **方式1：自动化测试（推荐）** ⭐⭐⭐⭐⭐

**一键完成所有测试：**

```bash
python run_100_samples_test.py
```

**这个脚本会自动：**
1. 运行 JSMA + SparseFool + Greedy（900个测试）
2. 运行 RandomSparse + PixelGrad（600个测试）
3. 生成完整分析报告
4. 显示详细进度和耗时

**预计时间：** 60-90分钟

**优点：**
- ✅ 完全自动化，无需人工干预
- ✅ 实时显示进度
- ✅ 自动生成报告
- ✅ 可以去做其他事情

---

### **方式2：分步执行（手动控制）**

如果你想分步执行，可以：

#### **步骤1：测试前3个方法**
```bash
python final_test_with_greedy.py
```
- 测试 JSMA, SparseFool, Greedy
- 3个模型 × 3个方法 × 100样本 = 900个测试
- 预计耗时：40-60分钟

#### **步骤2：测试后2个方法**
```bash
python test_new_2methods.py
```
- 测试 RandomSparse, PixelGrad
- 3个模型 × 2个方法 × 100样本 = 600个测试
- 预计耗时：20-30分钟

#### **步骤3：生成分析报告**
```bash
python analyze_all_5methods.py
```
- 汇总所有结果
- 生成图表和LaTeX表格
- 预计耗时：1-2分钟

---

## 📊 测试完成后会得到什么？

### **数据文件**（保存在 `results/complete_baseline/`）
```
resnet18_jsma.json
resnet18_sparsefool.json
resnet18_greedy.json
resnet18_randomsparse.json
resnet18_pixelgrad.json
vgg16_jsma.json
vgg16_sparsefool.json
vgg16_greedy.json
vgg16_randomsparse.json
vgg16_pixelgrad.json
mobilenetv2_jsma.json
mobilenetv2_sparsefool.json
mobilenetv2_greedy.json
mobilenetv2_randomsparse.json
mobilenetv2_pixelgrad.json
```

### **分析报告**（保存在 `results/paper_materials/`）
```
analysis_report_5methods.md      # 完整分析报告（英文）
latex_table_5methods.tex         # LaTeX表格（可直接用于论文）
asr_comparison_5methods.png      # ASR对比柱状图
asr_comparison_5methods.pdf      # PDF版本（高质量）
l0_comparison_5methods.png       # L0对比图
efficiency_scatter_5methods.png  # 效率散点图
asr_heatmap_5methods.png        # ASR热力图
```

---

## 🔍 测试完成后的检查清单

### **1. 数据完整性检查**
- [ ] 所有15个JSON文件都生成了吗？
- [ ] 每个文件大小是否合理（>10KB）？
- [ ] 没有报错或异常？

### **2. 结果合理性检查**
- [ ] ASR是否在合理范围（20%-95%）？
- [ ] L0是否在预期范围（3-8）？
- [ ] 不同方法之间是否有明显差异？
- [ ] RandomSparse的ASR是否最低？（这是预期的）

### **3. 与30样本结果对比**
预期变化：
- ✅ ASR的标准差应该变小
- ✅ 统计显著性应该更强（p值更小）
- ✅ 平均值应该相对稳定（±5%）

### **4. 图表质量检查**
- [ ] 图表是否清晰？
- [ ] 英文标签是否正确？
- [ ] PDF文件是否正常生成？
- [ ] 数据是否与表格一致？

---

## 📈 预期结果（基于30样本数据）

### **ASR对比**（预期）
| 方法 | ResNet18 | VGG16 | MobileNetV2 | 平均 |
|------|----------|-------|-------------|------|
| JSMA | ~83% | ~70% | ~93% | ~82% |
| SparseFool | ~73% | ~50% | ~93% | ~72% |
| Greedy | ~83% | ~73% | ~83% | ~80% |
| PixelGrad | ~37% | ~60% | ~50% | ~49% |
| RandomSparse | ~20% | ~23% | ~40% | ~28% |

### **关键发现**（预期保持）
1. ✅ RandomSparse显著差于其他方法
2. ✅ JSMA和Greedy最有效
3. ✅ 智能方法比随机方法提升 **76-195%**
4. ✅ MobileNetV2最容易攻击

---

## ⚠️ 可能遇到的问题

### **问题1：CUDA内存不足**
```python
RuntimeError: CUDA out of memory
```
**解决方案：**
- 关闭其他占用GPU的程序
- 或者暂时运行单个模型测试

### **问题2：测试时间太长**
**正常！** 100样本确实需要时间：
- JSMA最慢（~0.5s/样本）
- 总共1500个测试
- 60-90分钟是正常的

**建议：**
- 晚上运行，第二天查看结果
- 或者先测试单个模型验证

### **问题3：某个样本报错**
脚本会自动跳过失败的样本并继续，这是正常的。
少量失败（<5%）不影响结果。

---

## 🎯 完成后的下一步

测试完成并验证结果后，你将：

1. ✅ **完成 Week 1 Day 1-2** 🎉
2. 📝 查看 `analysis_report_5methods.md` 详细分析
3. 📊 检查所有生成的图表
4. 🎯 准备进入 **Week 1 Day 3-4：防御模型测试**

---

## 💡 小贴士

### **如果想先快速验证**
可以先用10个样本测试：
```python
# 临时修改配置
CONFIG['test_samples'] = 10
```
验证流程正常后，再改回100。

### **如果GPU资源紧张**
可以分批次运行：
```bash
# 先运行ResNet18
python final_test_with_greedy.py  # 修改代码只测试ResNet18

# 再运行VGG16
# ...以此类推
```

### **保存中间结果**
测试完成后，建议备份 `results/` 文件夹：
```bash
# 备份到其他位置，防止意外覆盖
```

---

## 🚀 准备好了吗？

**现在运行：**
```bash
python run_100_samples_test.py
```

**或者直接运行：**
```bash
python final_test_with_greedy.py
```

**祝测试顺利！** 💪

有任何问题随时问我！
















