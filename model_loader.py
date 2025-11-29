# model_loader.py
"""
多模型加载器
支持：ResNet18, VGG16, MobileNetV2, DenseNet121, ViT
"""

import torch
import torchvision.models as models


class ModelLoader:
    """统一的模型加载接口"""
    
    @staticmethod
    def load_model(model_name, num_classes=10, pretrained=True):
        """
        加载模型
        
        参数:
            model_name: 模型名称
            num_classes: 分类类别数
            pretrained: 是否使用预训练权重
        
        返回:
            model: 加载的模型
        """
        model_name = model_name.lower()
        
        if model_name == 'resnet18':
            if pretrained:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
            else:
                weights = None
            model = models.resnet18(weights=weights)
            model.fc = torch.nn.Linear(512, num_classes)
        
        elif model_name == 'resnet50':
            if pretrained:
                weights = models.ResNet50_Weights.IMAGENET1K_V1
            else:
                weights = None
            model = models.resnet50(weights=weights)
            model.fc = torch.nn.Linear(2048, num_classes)
        
        elif model_name == 'vgg16':
            if pretrained:
                weights = models.VGG16_Weights.IMAGENET1K_V1
            else:
                weights = None
            model = models.vgg16(weights=weights)
            model.classifier[6] = torch.nn.Linear(4096, num_classes)
        
        elif model_name == 'mobilenetv2':
            if pretrained:
                weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            else:
                weights = None
            model = models.mobilenet_v2(weights=weights)
            model.classifier[1] = torch.nn.Linear(1280, num_classes)
        
        elif model_name == 'densenet121':
            if pretrained:
                weights = models.DenseNet121_Weights.IMAGENET1K_V1
            else:
                weights = None
            model = models.densenet121(weights=weights)
            model.classifier = torch.nn.Linear(1024, num_classes)
        
        elif model_name == 'vit_b_16':
            if pretrained:
                weights = models.ViT_B_16_Weights.IMAGENET1K_V1
            else:
                weights = None
            model = models.vit_b_16(weights=weights)
            model.heads.head = torch.nn.Linear(768, num_classes)
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        return model.eval()
    
    @staticmethod
    def get_available_models():
        """获取所有可用的模型列表"""
        return [
            'resnet18',
            'resnet50',
            'vgg16',
            'mobilenetv2',
            'densenet121',
            'vit_b_16'
        ]
    
    @staticmethod
    def get_model_info(model_name):
        """获取模型信息"""
        info_dict = {
            'resnet18': {
                'name': 'ResNet-18',
                'params': '11.7M',
                'type': 'CNN'
            },
            'resnet50': {
                'name': 'ResNet-50',
                'params': '25.6M',
                'type': 'CNN'
            },
            'vgg16': {
                'name': 'VGG-16',
                'params': '138M',
                'type': 'CNN'
            },
            'mobilenetv2': {
                'name': 'MobileNetV2',
                'params': '3.5M',
                'type': 'CNN'
            },
            'densenet121': {
                'name': 'DenseNet-121',
                'params': '8.0M',
                'type': 'CNN'
            },
            'vit_b_16': {
                'name': 'ViT-B/16',
                'params': '86M',
                'type': 'Transformer'
            }
        }
        
        return info_dict.get(model_name.lower(), {'name': model_name, 'params': 'Unknown', 'type': 'Unknown'})


def get_experiment_models(num_classes=10, quick_test=False):
    """
    获取实验用的模型列表
    
    参数:
        num_classes: 分类类别数
        quick_test: 是否快速测试（只用2个模型）
    
    返回:
        models_dict: {model_name: model}
    """
    loader = ModelLoader()
    
    if quick_test:
        # 快速测试：只用2个模型
        model_names = ['resnet18', 'mobilenetv2']
    else:
        # 完整实验：5个模型（不包括ViT以节省时间）
        model_names = ['resnet18', 'vgg16', 'mobilenetv2', 'densenet121']
    
    models_dict = {}
    
    print("\n🔧 加载模型...")
    for model_name in model_names:
        try:
            model = loader.load_model(model_name, num_classes=num_classes, pretrained=True)
            models_dict[model_name] = model
            print(f"  ✅ {loader.get_model_info(model_name)['name']}")
        except Exception as e:
            print(f"  ❌ {model_name} 加载失败: {e}")
    
    return models_dict


# 使用示例
if __name__ == "__main__":
    print("🧪 测试模型加载器")
    
    loader = ModelLoader()
    
    # 显示所有可用模型
    print("\n可用模型:")
    for model_name in loader.get_available_models():
        info = loader.get_model_info(model_name)
        print(f"  - {info['name']} ({info['params']}, {info['type']})")
    
    # 加载一个模型测试
    print("\n加载 ResNet-18 测试...")
    model = loader.load_model('resnet18', num_classes=10)
    print(f"  模型类型: {type(model)}")
    print(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
        print(f"  输出形状: {y.shape}")
    
    print("\n✅ 测试完成！")

