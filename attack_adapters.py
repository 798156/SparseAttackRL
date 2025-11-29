#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
攻击方法适配器
统一不同攻击方法的接口，使其返回统一的格式
"""

import torch
import time
import numpy as np
from jsma_attack import jsma_attack
from sparsefool_attack import sparsefool_attack
from greedy_attack import greedy_attack
from pixel_gradient_attack import pixel_gradient_attack
from random_sparse_attack import random_sparse_attack_smart

def compute_metrics(original, adversarial, modified_pixels):
    """计算L0, L2等指标"""
    # L0: 修改的像素数
    if isinstance(modified_pixels, list):
        l0_norm = len(modified_pixels)
    else:
        l0_norm = modified_pixels
    
    # L2: 扰动的L2范数
    diff = (adversarial - original).cpu().numpy()
    l2_norm = np.linalg.norm(diff)
    
    return l0_norm, l2_norm

def jsma_attack_adapter(model, image, label, max_pixels=10, theta=0.2, max_iterations=100, device='cuda'):
    """JSMA适配器"""
    start_time = time.time()
    
    # 调用原始攻击（注意参数顺序：image, label, model）
    success, adv_image, modified_pixels = jsma_attack(
        image.squeeze(0),  # 去掉batch维度
        label.item(),
        model,
        max_pixels=max_pixels,
        theta=theta
    )
    
    elapsed_time = time.time() - start_time
    
    # 计算指标
    if success:
        l0_norm, l2_norm = compute_metrics(image, adv_image.unsqueeze(0), modified_pixels)
    else:
        l0_norm = l2_norm = 0
    
    info = {
        'l0_norm': l0_norm,
        'l2_norm': l2_norm,
        'time': elapsed_time,
        'iterations': len(modified_pixels) if modified_pixels else 0
    }
    
    return adv_image.unsqueeze(0) if success else image, success, info

def sparsefool_attack_adapter(model, image, label, max_iter=20, overshoot=0.02, lambda_=3.0, device='cuda'):
    """SparseFool适配器"""
    start_time = time.time()
    
    # 调用原始攻击
    success, adv_image, modified_pixels = sparsefool_attack(
        image.squeeze(0),
        label.item(),
        model,
        max_iterations=max_iter,
        lambda_=lambda_,
        overshoot=overshoot
    )
    
    elapsed_time = time.time() - start_time
    
    # 计算指标
    if success:
        l0_norm, l2_norm = compute_metrics(image, adv_image.unsqueeze(0), modified_pixels)
    else:
        l0_norm = l2_norm = 0
    
    info = {
        'l0_norm': l0_norm,
        'l2_norm': l2_norm,
        'time': elapsed_time,
        'iterations': max_iter
    }
    
    return adv_image.unsqueeze(0) if success else image, success, info

def greedy_attack_adapter(model, image, label, max_pixels=10, alpha=0.1, max_iterations=100, device='cuda'):
    """Greedy适配器"""
    start_time = time.time()
    
    # 调用原始攻击（使用step_size参数）
    success, adv_image, modified_pixels = greedy_attack(
        image.squeeze(0),
        label.item(),
        model,
        max_pixels=max_pixels,
        step_size=alpha
    )
    
    elapsed_time = time.time() - start_time
    
    # 计算指标
    if success:
        l0_norm, l2_norm = compute_metrics(image, adv_image.unsqueeze(0), modified_pixels)
    else:
        l0_norm = l2_norm = 0
    
    info = {
        'l0_norm': l0_norm,
        'l2_norm': l2_norm,
        'time': elapsed_time,
        'iterations': len(modified_pixels) if modified_pixels else 0
    }
    
    return adv_image.unsqueeze(0) if success else image, success, info

def pixel_gradient_attack_adapter(model, image, label, max_pixels=10, alpha=0.2, beta=0.9, device='cuda'):
    """PixelGrad适配器"""
    start_time = time.time()
    
    # 调用原始攻击
    success, adv_image, modified_pixels = pixel_gradient_attack(
        image.squeeze(0),
        label.item(),
        model,
        max_pixels=max_pixels,
        alpha=alpha,
        beta=beta
    )
    
    elapsed_time = time.time() - start_time
    
    # 计算指标
    if success:
        l0_norm, l2_norm = compute_metrics(image, adv_image.unsqueeze(0), modified_pixels)
    else:
        l0_norm = l2_norm = 0
    
    info = {
        'l0_norm': l0_norm,
        'l2_norm': l2_norm,
        'time': elapsed_time,
        'iterations': len(modified_pixels) if modified_pixels else 0
    }
    
    return adv_image.unsqueeze(0) if success else image, success, info

def random_sparse_attack_adapter(model, image, label, max_pixels=10, perturbation_size=0.2, max_attempts=50, device='cuda'):
    """RandomSparse适配器"""
    start_time = time.time()
    
    # 调用原始攻击
    success, adv_image, modified_pixels = random_sparse_attack_smart(
        image.squeeze(0),
        label.item(),
        model,
        max_pixels=max_pixels,
        perturbation_size=perturbation_size,
        max_attempts=max_attempts
    )
    
    elapsed_time = time.time() - start_time
    
    # 计算指标
    if success:
        l0_norm, l2_norm = compute_metrics(image, adv_image.unsqueeze(0), modified_pixels)
    else:
        l0_norm = l2_norm = 0
    
    info = {
        'l0_norm': l0_norm,
        'l2_norm': l2_norm,
        'time': elapsed_time,
        'iterations': max_attempts
    }
    
    return adv_image.unsqueeze(0) if success else image, success, info















