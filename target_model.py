# target_model.py
import torch
import torchvision
import os

def load_target_model(model_name="resnet18", weights=None, num_classes=10):
    """
    加载预训练模型，兼容新版本 torchvision
    
    ⚠️ 重要：如果使用CIFAR-10，需要先训练模型或加载训练好的权重
    """
    if model_name == "resnet18":
        # 创建模型
        model = torchvision.models.resnet18(weights=None)  # 从头开始
        model.fc = torch.nn.Linear(512, num_classes)
        
        # 尝试加载CIFAR-10训练好的权重
        cifar10_weights_path = 'cifar10_resnet18.pth'
        if os.path.exists(cifar10_weights_path):
            print(f"✅ 加载CIFAR-10训练的权重: {cifar10_weights_path}")
            model.load_state_dict(torch.load(cifar10_weights_path, map_location='cpu', weights_only=False))
        else:
            print("⚠️  警告：未找到CIFAR-10训练的权重！")
            print("   模型准确率会很低（~15%）")
            print("   请运行: python download_pretrained_cifar10.py")
            print(f"   或手动训练模型并保存为: {cifar10_weights_path}")

    elif model_name == "vgg16":
        model = torchvision.models.vgg16(weights=None)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
        
        vgg_weights_path = 'cifar10_vgg16.pth'
        if os.path.exists(vgg_weights_path):
            print(f"✅ 加载CIFAR-10训练的权重: {vgg_weights_path}")
            model.load_state_dict(torch.load(vgg_weights_path, map_location='cpu'))
        else:
            print("⚠️  警告：未找到CIFAR-10训练的VGG16权重！")

    elif model_name == "mobilenetv2":
        model = torchvision.models.mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(1280, num_classes)
        
        mobilenet_weights_path = 'cifar10_mobilenetv2.pth'
        if os.path.exists(mobilenet_weights_path):
            print(f"✅ 加载CIFAR-10训练的权重: {mobilenet_weights_path}")
            model.load_state_dict(torch.load(mobilenet_weights_path, map_location='cpu'))
        else:
            print("⚠️  警告：未找到CIFAR-10训练的MobileNetV2权重！")

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model.eval()

