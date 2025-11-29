#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加载已训练模型的辅助函数
"""

import torch
from model_loader import ModelLoader

def load_trained_model(model_name, checkpoint_path, device='cuda', num_classes=10):
    """
    加载已训练的模型
    
    参数:
        model_name: 模型名称 (resnet18, vgg16, mobilenetv2等)
        checkpoint_path: 检查点文件路径
        device: 设备 (cuda/cpu)
        num_classes: 分类类别数
    
    返回:
        model: 加载好权重的模型
    """
    # 创建模型
    loader = ModelLoader()
    model = loader.load_model(model_name, num_classes=num_classes, pretrained=False)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 兼容不同的checkpoint格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 移动到设备并设置为评估模式
    model = model.to(device)
    model.eval()
    
    return model















