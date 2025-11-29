"""
诊断脚本：验证ResNet18和VGG16模型是否真的不同
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

def load_model(model_name, device):
    """加载指定模型"""
    if model_name == 'ResNet18':
        model = torchvision.models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        model.load_state_dict(torch.load('cifar10_resnet18.pth', map_location=device, weights_only=False))
    
    elif model_name == 'VGG16':
        model = torchvision.models.vgg16(weights=None)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 10)
        model.load_state_dict(torch.load('cifar10_vgg16.pth', map_location=device, weights_only=False))
    
    model.to(device)
    model.eval()
    return model

def main():
    print("="*80)
    print("🔬 模型差异性验证")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  设备: {device}\n")
    
    # 加载数据
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # 加载模型
    print("📦 加载ResNet18...")
    resnet = load_model('ResNet18', device)
    print("📦 加载VGG16...")
    vgg = load_model('VGG16', device)
    
    # 测试相同样本的输出
    print("\n" + "="*80)
    print("🧪 测试: 相同输入，模型输出是否不同？")
    print("="*80)
    
    same_count = 0
    diff_count = 0
    
    np.random.seed(42)
    test_indices = np.random.choice(len(testset), 50, replace=False)
    
    print(f"\n测试 50 个随机样本...\n")
    
    for idx in test_indices:
        image, true_label = testset[idx]
        image_batch = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            resnet_output = resnet(image_batch)
            vgg_output = vgg(image_batch)
            
            resnet_pred = resnet_output.argmax(dim=1).item()
            vgg_pred = vgg_output.argmax(dim=1).item()
            
            # 获取预测概率
            resnet_prob = torch.softmax(resnet_output, dim=1)[0, resnet_pred].item()
            vgg_prob = torch.softmax(vgg_output, dim=1)[0, vgg_pred].item()
            
            if resnet_pred == vgg_pred:
                same_count += 1
            else:
                diff_count += 1
                print(f"  样本 {idx}: ResNet→{resnet_pred}({resnet_prob:.3f}), "
                      f"VGG→{vgg_pred}({vgg_prob:.3f}), 真实→{true_label}")
    
    print(f"\n{'='*80}")
    print("📊 统计结果:")
    print(f"{'='*80}")
    print(f"  预测相同: {same_count}/50 = {same_count/50*100:.1f}%")
    print(f"  预测不同: {diff_count}/50 = {diff_count/50*100:.1f}%")
    
    if diff_count == 0:
        print("\n❌ 严重问题：两个模型对所有样本的预测完全相同！")
        print("   可能原因：模型文件实际上是相同的，或加载出错")
    elif diff_count > 0 and diff_count < 5:
        print("\n⚠️  警告：预测差异很小，可能存在问题")
    else:
        print("\n✅ 正常：两个模型确实不同")
        print(f"   差异率 {diff_count/50*100:.1f}% 是合理的")
    
    # 验证选择的样本是否不同
    print(f"\n{'='*80}")
    print("🧪 测试: 样本选择是否不同？")
    print(f"{'='*80}")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 为ResNet选择样本
    resnet_samples = []
    for idx in range(len(testset)):
        if len(resnet_samples) >= 30:
            break
        image, label = testset[idx]
        image_batch = image.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = resnet(image_batch).argmax(dim=1).item()
        if pred == label:
            resnet_samples.append(idx)
    
    # 为VGG选择样本（重置随机种子）
    np.random.seed(42)
    torch.manual_seed(42)
    
    vgg_samples = []
    for idx in range(len(testset)):
        if len(vgg_samples) >= 30:
            break
        image, label = testset[idx]
        image_batch = image.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = vgg(image_batch).argmax(dim=1).item()
        if pred == label:
            vgg_samples.append(idx)
    
    print(f"\nResNet18 选中的前10个样本ID: {resnet_samples[:10]}")
    print(f"VGG16 选中的前10个样本ID: {vgg_samples[:10]}")
    
    same_samples = set(resnet_samples) == set(vgg_samples)
    common_samples = len(set(resnet_samples) & set(vgg_samples))
    
    print(f"\n共同样本数: {common_samples}/30")
    
    if same_samples:
        print("⚠️  警告：两个模型选中了完全相同的样本！")
        print("   这解释了为什么测试结果相同")
        print("   建议：使用预先选定的样本集，而不是按模型选择")
    else:
        print(f"✅ 正常：样本集有 {30-common_samples} 个不同")
    
    print(f"\n{'='*80}")
    print("💡 诊断建议:")
    print(f"{'='*80}")
    
    if diff_count == 0:
        print("1. ❌ 模型本身可能有问题，检查训练过程")
    elif same_samples and diff_count > 0:
        print("1. ⚠️  虽然选中了相同样本，但模型预测不同")
        print("2. 💡 测试结果完全相同可能是巧合，需要更多样本验证")
        print("3. 🔧 建议使用固定的测试样本集（基于真实标签选择）")
    else:
        print("1. ✅ 模型和样本选择都正常")
        print("2. ❓ 如果测试结果仍然相同，可能是随机性导致的巧合")
        print("3. 🔧 建议增加测试样本数量到100+")

if __name__ == "__main__":
    main()
















