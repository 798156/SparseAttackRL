"""
诊断JSMA和SparseFool实现
验证参数影响和实现正确性
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from jsma_attack import jsma_attack
from sparsefool_attack import sparsefool_attack
import matplotlib.pyplot as plt

def load_model(model_name='ResNet18'):
    """加载模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_name == 'ResNet18':
        model = torchvision.models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        model.load_state_dict(torch.load('cifar10_resnet18.pth', map_location=device, weights_only=False))
    
    model.to(device)
    model.eval()
    return model, device

def load_test_data():
    """加载测试数据"""
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    return testset

def test_jsma_with_different_theta():
    """测试JSMA在不同theta下的表现"""
    print("\n" + "="*80)
    print("🧪 测试JSMA - 不同theta参数的影响")
    print("="*80)
    
    model, device = load_model()
    testset = load_test_data()
    
    # 选择10个正确分类的样本
    selected_samples = []
    for idx in range(len(testset)):
        if len(selected_samples) >= 10:
            break
        image, label = testset[idx]
        image_batch = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_batch)
            pred = output.argmax(dim=1).item()
        if pred == label:
            selected_samples.append((idx, image, label))
    
    print(f"✅ 选择了 {len(selected_samples)} 个样本\n")
    
    # 测试不同的theta值
    theta_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    results = {}
    for theta in theta_values:
        print(f"\n{'='*40}")
        print(f"测试 theta={theta}")
        print(f"{'='*40}")
        
        success_count = 0
        l0_values = []
        
        for i, (idx, image, label) in enumerate(selected_samples):
            success, adv_image, modified_pixels = jsma_attack(
                image=image,
                label=label,
                model=model,
                max_pixels=10,
                theta=theta
            )
            
            if success:
                success_count += 1
                # 计算实际修改的像素数（确保在CPU）
                adv_cpu = adv_image.detach().cpu()
                img_cpu = image.cpu() if image.device.type != 'cpu' else image
                diff = (adv_cpu - img_cpu).abs()
                l0 = (diff.sum(dim=0) > 1e-5).sum().item()
                l0_values.append(l0)
                print(f"  样本{i}: ✅ 成功, L0={l0}")
            else:
                print(f"  样本{i}: ❌ 失败")
        
        asr = success_count / len(selected_samples) * 100
        avg_l0 = np.mean(l0_values) if l0_values else 0
        
        results[theta] = {
            'ASR': asr,
            'avg_L0': avg_l0,
            'success_count': success_count
        }
        
        print(f"\n📊 theta={theta}:")
        print(f"  ASR: {success_count}/{len(selected_samples)} = {asr:.1f}%")
        print(f"  平均L0: {avg_l0:.2f}")
    
    # 汇总
    print(f"\n{'='*80}")
    print("📊 JSMA - Theta参数影响汇总")
    print(f"{'='*80}\n")
    print(f"{'Theta':<10} {'ASR':<10} {'平均L0'}")
    print("-"*40)
    for theta in theta_values:
        r = results[theta]
        print(f"{theta:<10} {r['ASR']:<10.1f} {r['avg_L0']:.2f}")
    
    return results

def test_sparsefool_with_different_params():
    """测试SparseFool在不同参数下的表现"""
    print("\n" + "="*80)
    print("🧪 测试SparseFool - 不同参数的影响")
    print("="*80)
    
    model, device = load_model()
    testset = load_test_data()
    
    # 选择10个正确分类的样本
    selected_samples = []
    for idx in range(len(testset)):
        if len(selected_samples) >= 10:
            break
        image, label = testset[idx]
        image_batch = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_batch)
            pred = output.argmax(dim=1).item()
        if pred == label:
            selected_samples.append((idx, image, label))
    
    print(f"✅ 选择了 {len(selected_samples)} 个样本\n")
    
    # 测试不同的max_iterations
    max_iter_values = [10, 20, 50, 100]
    
    results = {}
    for max_iter in max_iter_values:
        print(f"\n{'='*40}")
        print(f"测试 max_iterations={max_iter}")
        print(f"{'='*40}")
        
        success_count = 0
        l0_values = []
        
        for i, (idx, image, label) in enumerate(selected_samples):
            success, adv_image, modified_pixels = sparsefool_attack(
                image=image,
                label=label,
                model=model,
                max_iterations=max_iter,
                lambda_=3.0
            )
            
            if success:
                success_count += 1
                # 计算实际修改的像素数（确保在CPU）
                adv_cpu = adv_image.detach().cpu()
                img_cpu = image.cpu() if image.device.type != 'cpu' else image
                diff = (adv_cpu - img_cpu).abs()
                l0 = (diff.sum(dim=0) > 1e-5).sum().item()
                l0_values.append(l0)
                print(f"  样本{i}: ✅ 成功, L0={l0}")
            else:
                print(f"  样本{i}: ❌ 失败")
        
        asr = success_count / len(selected_samples) * 100
        avg_l0 = np.mean(l0_values) if l0_values else 0
        
        results[max_iter] = {
            'ASR': asr,
            'avg_L0': avg_l0,
            'success_count': success_count
        }
        
        print(f"\n📊 max_iter={max_iter}:")
        print(f"  ASR: {success_count}/{len(selected_samples)} = {asr:.1f}%")
        print(f"  平均L0: {avg_l0:.2f}")
    
    # 汇总
    print(f"\n{'='*80}")
    print("📊 SparseFool - 参数影响汇总")
    print(f"{'='*80}\n")
    print(f"{'MaxIter':<10} {'ASR':<10} {'平均L0'}")
    print("-"*40)
    for max_iter in max_iter_values:
        r = results[max_iter]
        print(f"{max_iter:<10} {r['ASR']:<10.1f} {r['avg_L0']:.2f}")
    
    return results

def visualize_attack_effect():
    """可视化攻击效果"""
    print("\n" + "="*80)
    print("🎨 可视化攻击效果")
    print("="*80)
    
    model, device = load_model()
    testset = load_test_data()
    
    # 找一个成功的样本
    for idx in range(100):
        image, label = testset[idx]
        image_batch = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_batch)
            pred = output.argmax(dim=1).item()
        
        if pred == label:
            # 尝试JSMA攻击
            success_jsma, adv_jsma, _ = jsma_attack(
                image, label, model, max_pixels=10, theta=1.0
            )
            
            # 尝试SparseFool攻击
            success_sf, adv_sf, _ = sparsefool_attack(
                image, label, model, max_iterations=20
            )
            
            if success_jsma or success_sf:
                print(f"✅ 找到可视化样本: idx={idx}")
                print(f"  原始标签: {label}")
                
                # 计算L0（确保在CPU）
                if success_jsma:
                    diff_jsma = (adv_jsma.cpu() - image.cpu()).abs()
                    l0_jsma = (diff_jsma.sum(dim=0) > 1e-5).sum().item()
                    with torch.no_grad():
                        pred_jsma = model(adv_jsma.unsqueeze(0).to(device)).argmax(dim=1).item()
                    print(f"  JSMA: 成功, L0={l0_jsma}, 预测={pred_jsma}")
                
                if success_sf:
                    diff_sf = (adv_sf.cpu() - image.cpu()).abs()
                    l0_sf = (diff_sf.sum(dim=0) > 1e-5).sum().item()
                    with torch.no_grad():
                        pred_sf = model(adv_sf.unsqueeze(0).to(device)).argmax(dim=1).item()
                    print(f"  SparseFool: 成功, L0={l0_sf}, 预测={pred_sf}")
                
                break

def main():
    """主函数"""
    print("\n" + "="*80)
    print("🔬 对抗攻击实现诊断")
    print("="*80)
    print("\n目标:")
    print("  1. 测试不同参数对ASR和L0的影响")
    print("  2. 找到合适的统一参数配置")
    print("  3. 验证实现正确性\n")
    
    # 测试JSMA
    jsma_results = test_jsma_with_different_theta()
    
    # 测试SparseFool
    sf_results = test_sparsefool_with_different_params()
    
    # 可视化
    visualize_attack_effect()
    
    # 推荐参数
    print(f"\n{'='*80}")
    print("💡 参数推荐")
    print(f"{'='*80}\n")
    
    print("基于诊断结果，推荐参数配置：\n")
    
    # 找到ASR最接近50-70%的theta
    target_asr = 60
    best_theta = min(jsma_results.keys(), 
                    key=lambda t: abs(jsma_results[t]['ASR'] - target_asr))
    
    print(f"JSMA:")
    print(f"  max_pixels: 10")
    print(f"  theta: {best_theta} (ASR={jsma_results[best_theta]['ASR']:.1f}%, L0={jsma_results[best_theta]['avg_L0']:.2f})")
    
    best_iter = min(sf_results.keys(),
                   key=lambda i: abs(sf_results[i]['ASR'] - target_asr))
    
    print(f"\nSparseFool:")
    print(f"  max_iterations: {best_iter} (ASR={sf_results[best_iter]['ASR']:.1f}%, L0={sf_results[best_iter]['avg_L0']:.2f})")
    print(f"  lambda_: 3.0")
    
    print(f"\n{'='*80}")
    print("🎉 诊断完成！")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

